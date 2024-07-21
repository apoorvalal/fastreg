import jax.numpy as np
from jax.scipy.special import gammaln
from jax.lax import logistic
from jax.numpy import exp

import pandas as pd

import jax
import jax.numpy as np
import jax.numpy.linalg as la
from jax.tree_util import tree_flatten, tree_leaves, tree_map
from .tools import block_inverse, maybe_diag, atleast_2d, hstack



# clamped log
def log(x, ε=1e-7):
    return np.log(np.maximum(ε, x))


# loss functions (only parameter relevant terms)
def binary_loss(yh, y):
    return y * log(yh) + (1 - y) * log(1 - yh)


def poisson_loss(yh, y):
    return y * log(yh) - yh


def negbin_loss(r, yh, y):
    return (
        gammaln(r + y) - gammaln(r) + r * log(r) + y * log(yh) - (r + y) * log(r + yh)
    )


def lstsq_loss(yh, y):
    return -((y - yh) ** 2)


def normal_loss(p, yh, y):
    lsigma2 = p["lsigma2"]
    sigma2 = np.exp(lsigma2)
    like = -0.5 * lsigma2 + 0.5 * lstsq_loss(yh, y) / sigma2
    return like


losses = {
    "logit": lambda p, d, yh, y: binary_loss(logistic(yh), y),
    "poisson": lambda p, d, yh, y: poisson_loss(exp(yh), y),
    "negbin": lambda p, d, yh, y: negbin_loss(exp(p["lr"]), exp(yh), y),
    "normal": lambda p, d, yh, y: normal_loss(p, yh, y),
    "lognorm": lambda p, d, yh, y: normal_loss(p, yh, log(y)),
    "lstsq": lambda p, d, yh, y: lstsq_loss(yh, y),
    "loglstsq": lambda p, d, yh, y: lstsq_loss(yh, log(y)),
}


def ensure_loss(s):
    if type(s) is str:
        return losses[s]
    else:
        return s


# loss function modifiers
def zero_inflate(like0, clip_like=20.0, key="lpzero"):
    like0 = ensure_loss(like0)

    def like(p, d, yh, y):
        pzero = logistic(p[key])
        blike = np.clip(like0(p, d, yh, y), a_max=clip_like)
        zlike = log(pzero + (1 - pzero) * exp(blike))
        plike = log(1 - pzero) + blike
        return np.where(y == 0, zlike, plike)

    return like


def add_offset(like0, key="offset"):
    like0 = ensure_loss(like0)

    def like(p, d, yh, y):
        yh1 = d[key] + yh
        return like0(p, d, yh1, y)

    return like

##
## batching it, pytree style
##


class DataLoader:
    def __init__(self, data, batch_size=None):
        # robust input handling
        if type(data) is pd.DataFrame:
            data = data.to_dict("series")

        # note that tree_map seems to drop None valued leaves
        self.data = tree_map(
            lambda x: np.array(x) if type(x) is not np.ndarray else x, data
        )

        # validate shapes
        shapes = set([d.shape[0] for d in tree_leaves(self.data)])
        if len(shapes) > 1:
            raise Exception("All data series must have first dimension size")

        # store for iteration
        (self.data_size,) = shapes
        self.batch_size = batch_size

    def __call__(self, batch_size=None):
        yield from self.iterate(batch_size)

    def __iter__(self):
        yield from self.iterate()

    def iterate(self, batch_size=None):
        # round off data size to batch_size multiple
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_batches = max(1, self.data_size // batch_size)
        round_size = batch_size * num_batches

        # yield successive tree batches
        for i in range(0, round_size, batch_size):
            yield tree_map(lambda d: d[i : i + batch_size], self.data)


# ignore batch_size and use entire dataset
class OneLoader:
    def __init__(self, data, batch_size=None):
        self.data = data

    def __call__(self, batch_size=None):
        yield from self.iterate()

    def __iter__(self):
        yield from self.iterate()

    def iterate(self, batch_size=None):
        yield self.data



##
## block matrices
##


def chunks(v, n):
    return [v[i : i + n] for i in range(0, len(v), n)]


def block_matrix(tree, dims, size):
    blocks = chunks(tree_leaves(tree), size)
    mat = np.block(
        [[atleast_2d(x, axis=int(d > 0)) for d, x in zip(dims, row)] for row in blocks]
    )
    return mat


def block_unpack(mat, tree, sizes):
    part = np.cumsum(np.array(sizes))
    block = lambda x, axis: tree.unflatten(np.split(x, part, axis=axis)[:-1])
    tree = tree_map(lambda x: block(x, 1), block(mat, 0))
    return tree


# we need par to know the inner shape
def tree_matfun(fun, mat, par):
    # get param configuration
    par_flat, par_tree = tree_flatten(par)
    par_sizs = [np.size(p) for p in par_flat]
    par_dims = [np.ndim(p) for p in par_flat]
    K = len(par_flat)

    # invert hessian for stderrs
    tmat = block_matrix(mat, par_dims, K)
    fmat = fun(tmat)
    fout = block_unpack(fmat, par_tree, par_sizs)

    return fout


def tree_batch_reduce(batch_fun, loader, agg_fun=np.add):
    total = None
    for b, batch in enumerate(loader):
        f_batch = batch_fun(batch)
        if total is None:
            total = f_batch
        else:
            total = tree_map(agg_fun, total, f_batch)
    return total


def tree_outer(tree):
    return tree_map(lambda x: tree_map(lambda y: x.T @ y, tree), tree)


def tree_outer_flat(tree):
    tree1, vec = dict_popoff(tree, "hdfe")
    leaves = [atleast_2d(l) for l in tree_leaves(tree1)]
    mat = np.hstack(leaves)
    A = mat.T @ mat
    B = mat.T @ vec
    C = vec.T @ mat
    d = np.sum(vec * vec, axis=0)
    return A, B, C, d


def dict_popoff(d, s):
    if s in d:
        return {k: v for k, v in d.items() if k != s}, d[s]
    else:
        return d, None


def tree_fisher(gv_fun, params, loader):
    # accumulate outer product
    fish = tree_batch_reduce(lambda b: tree_outer(gv_fun(params, b)), loader)

    # invert fisher matrix
    sigma = tree_matfun(la.inv, fish, params)

    return sigma


def diag_fisher(gv_fun, params, loader):
    # compute hessian inverse by block
    A, B, C, d = tree_batch_reduce(lambda b: tree_outer_flat(gv_fun(params, b)), loader)
    psig, hsig = block_inverse(A, B, C, d, inv=la.inv)

    # unpack into tree
    par0, _ = dict_popoff(params, "hdfe")
    par0_flat, par0_tree = tree_flatten(par0)
    par0_sizs = [np.size(p) for p in par0_flat]
    sigma = block_unpack(psig, par0_tree, par0_sizs)
    sigma["hdfe"] = hsig

    return sigma


# just get mean and var vectors
def flatten_output(beta, sigma):
    beta_reals = beta["reals"]
    beta_categ = hstack(beta["categ"].values())

    sigma_reals = (
        maybe_diag(sigma["reals"]["reals"]) if sigma["reals"] is not None else None
    )
    sigma_categ = hstack(
        [maybe_diag(sigma["categ"][c]["categ"][c]) for c in sigma["categ"]]
    )

    beta_vec = hstack([beta_reals, beta_categ])
    sigma_vec = hstack([sigma_reals, sigma_categ])

    return beta_vec, sigma_vec

