import pandas as pd
from operator import and_, add

import jax
import jax.numpy as np
from jax.tree_util import tree_map, tree_reduce
import optax

from .tools import chainer
from .formula import (
    parse_tuple,
    ensure_formula,
    categorize,
    is_categorical,
    Categ,
    Formula,
    O,
)
from .trees import design_tree
from .summary import param_table
from .utils import (
    losses,
    zero_inflate,
    OneLoader,
    DataLoader,
    tree_matfun,
    tree_batch_reduce,
    tree_fisher,
    diag_fisher,
    flatten_output,
)


##
## optimizers
##


# figure out burn-in - cosine decay
def lr_schedule(eta, epochs, boost=10.0, burn=0.15):
    burn = int(burn * epochs) if type(burn) is float else burn

    def get_lr(ep):
        decay = np.clip(ep / burn, 0, 1)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay))
        return eta * (1 + coeff * (boost - 1))

    return get_lr


# adam optimizer with initial boost + cosine decay
def adam(
    vg_fun,
    loader,
    params0,
    epochs=10,
    eta=0.005,
    beta1=0.9,
    beta2=0.99,
    eps=1e-8,
    xtol=1e-4,
    ftol=1e-5,
    boost=10.0,
    burn=0.4,
    disp=None,
):
    get_lr = lr_schedule(eta, epochs, boost=boost, burn=burn)

    # parameter info
    params = tree_map(np.array, params0)
    avg_loss = -np.inf

    # track rms gradient
    m = tree_map(np.zeros_like, params)
    v = tree_map(np.zeros_like, params)

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch, tot_batch = 0.0, 0, 0
        agg_grad = tree_map(np.zeros_like, params0)
        last_par, last_loss = params, avg_loss

        # iterate over batches
        for batch in loader:
            # compute gradients
            loss, grad = vg_fun(params, batch)

            # check for any nans
            lnan = np.isnan(loss)
            gnan = tree_reduce(and_, tree_map(lambda g: np.isnan(g).any(), grad))
            if lnan or gnan:
                print("Encountered nans!")
                return params, None

            # implement next step
            m = tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, m, grad)
            v = tree_map(lambda v, g: beta2 * v + (1 - beta2) * g**2, v, grad)

            # update with adjusted values
            lr = get_lr(ep)
            mhat = tree_map(lambda m: m / (1 - beta1 ** (tot_batch + 1)), m)
            vhat = tree_map(lambda v: v / (1 - beta2 ** (tot_batch + 1)), v)
            params = tree_map(
                lambda p, m, v: p + lr * m / (np.sqrt(v) + eps), params, mhat, vhat
            )

            # compute statistics
            agg_loss += loss
            agg_grad = tree_map(add, agg_grad, grad)
            agg_batch += 1
            tot_batch += 1

        # compute stats
        avg_loss = agg_loss / agg_batch
        avg_grad = tree_map(lambda x: x / agg_batch, agg_grad)
        abs_grad = tree_reduce(
            np.maximum, tree_map(lambda x: np.max(np.abs(x)), avg_grad)
        )
        par_diff = tree_reduce(
            np.maximum,
            tree_map(lambda p1, p2: np.max(np.abs(p1 - p2)), params, last_par),
        )
        loss_diff = np.abs(avg_loss - last_loss)

        # display output
        if disp is not None:
            disp(ep, avg_loss, abs_grad, par_diff, loss_diff, params)

        # check converge
        if par_diff < xtol and loss_diff < ftol:
            break

    # show final result
    if disp is not None:
        disp(ep, avg_loss, abs_grad, par_diff, loss_diff, params, final=True)

    return params


# adam optimizer with cosine burn in
def adam_cosine(learn=1e-2, boost=5.0, burn=0.3, epochs=None, **kwargs):
    burn = int(burn * epochs) if type(burn) is float else burn
    schedule = optax.cosine_decay_schedule(boost * learn, burn, alpha=1 / boost)
    return optax.chain(optax.scale_by_adam(**kwargs), optax.scale_by_schedule(schedule))


# adam optimizer with initial boost + cosine decay
def optax_wrap(
    vg_fun,
    loader,
    params0,
    optimizer=None,
    epochs=10,
    xtol=1e-4,
    ftol=1e-5,
    gtol=1e-4,
    disp=None,
    **kwargs,
):
    # default optimizer
    if optimizer is None:
        optimizer = adam_cosine(epochs=epochs, **kwargs)

    # initialize optimizer
    params = tree_map(np.array, params0)
    state = optimizer.init(params)

    # start at bottom
    avg_loss = -np.inf

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch, tot_batch = 0.0, 0, 0
        agg_grad = tree_map(np.zeros_like, params0)
        last_par, last_loss = params, avg_loss

        # iterate over batches
        for batch in loader:
            # compute gradients
            loss, grad = vg_fun(params, batch)

            # check for any nans
            lnan = np.isnan(loss)
            gnan = tree_reduce(and_, tree_map(lambda g: np.isnan(g).any(), grad))
            if lnan or gnan:
                print("Encountered nans!")
                return params, None

            # update with adjusted values
            updates, state = optimizer.update(grad, state, params)
            params = optax.apply_updates(params, updates)

            # compute statistics
            agg_loss += loss
            agg_grad = tree_map(add, agg_grad, grad)
            agg_batch += 1
            tot_batch += 1

        # compute stats
        avg_loss = agg_loss / agg_batch
        avg_grad = tree_map(lambda x: x / agg_batch, agg_grad)
        abs_grad = tree_reduce(
            np.maximum, tree_map(lambda x: np.max(np.abs(x)), avg_grad)
        )
        par_diff = tree_reduce(
            np.maximum,
            tree_map(lambda p1, p2: np.max(np.abs(p1 - p2)), params, last_par),
        )
        loss_diff = np.abs(avg_loss - last_loss)

        # display output
        if disp is not None:
            disp(ep, avg_loss, abs_grad, par_diff, loss_diff, params)

        # check converge
        if abs_grad < gtol and par_diff < xtol:
            break

    # show final result
    if disp is not None:
        disp(ep, avg_loss, abs_grad, par_diff, loss_diff, params, final=True)

    return params


##
## estimation
##


# maximum likelihood using jax - this expects a mean log likelihood
def maxlike(
    model=None,
    params=None,
    data=None,
    stderr=False,
    optim=adam,
    batch_size=32768,
    backend="cpu",
    **kwargs,
):
    # get model gradients
    vg_fun = jax.jit(jax.value_and_grad(model), backend=backend)

    # simple non-batched loader
    BatchLoader = OneLoader if batch_size is None else DataLoader
    loader = BatchLoader(data)

    # maximize likelihood
    params1 = optim(vg_fun, loader, params, **kwargs)

    if not stderr:
        return params1, None

    # get model hessian
    h_fun = jax.jit(jax.hessian(model), backend=backend)

    # compute standard errors
    hess = tree_batch_reduce(lambda b: h_fun(params, b), loader)
    fish = tree_matfun(np.linalg.inv, hess, params)
    omega = tree_map(lambda x: -x, fish)

    return params1, omega


# maximum likelihood using jax - this expects a mean log likelihood
# the assumes the data is batchable, which usually means panel-like
# a toplevel hdfe variable is treated special-like in diag_fisher
def maxlike_panel(
    model=None,
    params=None,
    data=None,
    vg_fun=None,
    stderr=True,
    optim=adam,
    batch_size=32768,
    backend="cpu",
    **kwargs,
):
    # compute gradient for optim
    vg_fun = jax.jit(jax.value_and_grad(model), backend=backend)

    # set up batching
    BatchLoader = OneLoader if batch_size is None else DataLoader
    loader = BatchLoader(data, batch_size)

    # maximize likelihood
    params1 = optim(vg_fun, loader, params, **kwargs)

    # just point estimates
    if not stderr:
        return params1, None

    # get vectorized gradient
    gv_fun = jax.jit(jax.vmap(jax.grad(model), (None, 0), 0), backend=backend)

    # compute standard errors
    if "hdfe" in params:
        sigma = diag_fisher(gv_fun, params1, loader)
    else:
        sigma = tree_fisher(gv_fun, params1, loader)

    return params1, sigma


# make a glm model with a particular loss
def glm_model(loss, hdfe=None):
    if type(loss) is str:
        loss = losses[loss]

    # evaluator function
    def model(par, dat):
        # load in data and params
        ydat, rdat, cdat, odat = dat["ydat"], dat["rdat"], dat["cdat"], dat["odat"]
        reals, categ = par["reals"], par["categ"]
        if hdfe is not None:
            categ[hdfe] = par.pop("hdfe")

        # evaluate linear predictor
        pred = odat
        if rdat is not None:
            pred += rdat @ reals
        for i, c in enumerate(categ):
            cidx = cdat.T[i]  # needed for vmap to work
            pred += np.where(cidx >= 0, categ[c][cidx], 0.0)  # -1 means drop

        # compute average likelihood
        like = loss(par, dat, pred, ydat)
        return np.mean(like)

    return model


# default glm specification
def glm(
    y=None,
    x=None,
    formula=None,
    hdfe=None,
    data=None,
    extra={},
    raw={},
    offset=None,
    model=None,
    loss=None,
    stderr=True,
    display=True,
    epochs=None,
    per=None,
    output="table",
    **kwargs,
):
    # convert to formula system
    y, x = ensure_formula(x=x, y=y, formula=formula)

    # add in hdfe if needed
    if hdfe is not None:
        c_hdfe = parse_tuple(hdfe, convert=Categ)
        x += c_hdfe
        hdfe = c_hdfe.name()

    # add in raw data with offset special case
    if offset is None:
        offset = O

    # get all data in tree form
    formulify = lambda ts: Formula(*ts) if len(ts) > 0 else None
    c, r = map(formulify, categorize(is_categorical, x))
    tree = {"ydat": y, "rdat": r, "cdat": c, "odat": offset, **raw}
    nam, dat = design_tree(tree, data=data)

    # handle no reals/categ case
    if tree["cdat"] is None:
        nam["cdat"] = {}
    if tree["rdat"] is None:
        nam["rdat"] = []

    # choose number of epochs
    N = len(dat["ydat"])
    epochs = max(1, 200_000_000 // N) if epochs is None else epochs
    per = max(1, epochs // 5) if per is None else per

    # create model if needed
    if model is None:
        model = glm_model(loss, hdfe=hdfe)

    # displayer
    def disp0(e, l, g, x, f, p, final=False):
        if e % per == 0 or final:
            reals, categ = p["reals"], p["categ"]
            if hdfe is not None:
                categ = categ.copy()
                categ[hdfe] = p["hdfe"]
            μr = np.mean(reals) if reals is not None else np.nan
            μc = np.mean(np.array([np.mean(c) for c in categ.values()]))
            print(
                f"[{e:3d}] ℓ={l:.5f}, g={g:.5f}, Δβ={x:.5f}, Δℓ={f:.5f}, μR={μr:.5f}, μC={μc:.5f}"
            )

    disp = disp0 if display else None

    # organize data and initial params
    preals = np.zeros(len(nam["rdat"])) if len(nam["rdat"]) > 0 else None
    pcateg = {c: np.zeros(len(ls)) for c, ls in nam["cdat"].items()}
    params = {"reals": preals, "categ": pcateg, **extra}
    if hdfe is not None:
        params["hdfe"] = params["categ"].pop(hdfe)

    # estimate model
    beta, sigma = maxlike_panel(
        model=model,
        params=params,
        data=dat,
        stderr=stderr,
        disp=disp,
        epochs=epochs,
        **kwargs,
    )

    # splice in hdfe results
    if hdfe is not None:
        beta["categ"][hdfe] = beta.pop("hdfe")
        if stderr:
            sigma["categ"][hdfe] = {"categ": {hdfe: sigma.pop("hdfe")}}

    # return requested info
    if output == "table":
        y_name = nam["ydat"]
        x_names = nam["rdat"] + chainer(nam["cdat"].values())
        beta_vec, sigma_vec = flatten_output(beta, sigma)
        return param_table(beta_vec, y_name, x_names, sigma=sigma_vec)
    elif output == "dict":
        names = {"reals": nam["rdat"], "categ": nam["cdat"]}
        return names, beta, sigma


# logit regression
def logit(y=None, x=None, data=None, **kwargs):
    return glm(y=y, x=x, data=data, loss="logit", **kwargs)


# poisson regression
def poisson(y=None, x=None, data=None, **kwargs):
    return glm(y=y, x=x, data=data, loss="poisson", **kwargs)


# zero inflated poisson regression
def poisson_zinf(y=None, x=None, data=None, clip_like=20.0, **kwargs):
    loss = zero_inflate(losses["poisson"], clip_like=clip_like)
    extra = {"lpzero": 0.0}
    return glm(y=y, x=x, data=data, loss=loss, extra=extra, **kwargs)


# negative binomial regression
def negbin(y=None, x=None, data=None, **kwargs):
    extra = {"lr": 0.0}
    return glm(y=y, x=x, data=data, loss="negbin", extra=extra, **kwargs)


# zero inflated poisson regression
def negbin_zinf(y=None, x=None, data=None, clip_like=20.0, **kwargs):
    loss = zero_inflate(losses["negbin"], clip_like=clip_like)
    extra = {"lpzero": 0.0, "lr": 0.0}
    return glm(y=y, x=x, data=data, loss=loss, extra=extra, **kwargs)


# implement ols with full sigma
def gols(y=None, x=None, data=None, **kwargs):
    extra = {"lsigma2": 0.0}
    return glm(y=y, x=x, data=data, loss="normal", extra=extra, **kwargs)
