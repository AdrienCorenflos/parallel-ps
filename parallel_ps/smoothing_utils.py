from functools import partial
from typing import Callable, Any

import jax
from jax import numpy as jnp, tree_map, vmap
from jax.scipy.special import logsumexp

from parallel_ps.base import split_batched_and_static_params, rejoin_batched_and_static_params, PyTree


def make_log_weight_fn_and_params_inputs(nut_log_potential, nut_params, nut_batched_flag,
                                         Mt_log_potential, Mt_params, Mt_batched_flag,
                                         Gt_log_potential, Gt_params, Gt_batched_flag,
                                         T: int):
    """
    This function organises parameters and creates the weight function used for stitching in the paper.
    """
    # The way we build this function is so that if the parameters are static for the model we don't duplicate them to
    # be passed to the divide and conquer map.

    batched_nut_params, static_nut_params = split_batched_and_static_params(nut_params,
                                                                            nut_batched_flag)
    batched_Mt_params, static_Mt_params = split_batched_and_static_params(Mt_params,
                                                                          Mt_batched_flag)
    batched_Gt_params, static_Gt_params = split_batched_and_static_params(Gt_params,
                                                                          Gt_batched_flag)

    # We pad as the first index is not used but the shapes need to be consistent.
    def _pad_params(batch_param):
        ndim = batch_param.ndim
        t = batch_param.shape[0]
        return jnp.pad(batch_param, [(T - t, 0)] + [(0, 0)] * (ndim - 1), constant_values=0)

    batched_nut_params = jax.tree_map(_pad_params, batched_nut_params)
    batched_Mt_params = jax.tree_map(_pad_params, batched_Mt_params)
    batched_Gt_params = jax.tree_map(_pad_params, batched_Gt_params)

    params = batched_nut_params, batched_Mt_params, batched_Gt_params

    @jax.jit
    def log_weight_function(x_t_1, x_t, params_t):
        nut_params_t, Mt_params_t, Gt_params_t = params_t

        nut_params_t = rejoin_batched_and_static_params(nut_params_t, static_nut_params,
                                                        nut_batched_flag)
        Mt_params_t = rejoin_batched_and_static_params(Mt_params_t, static_Mt_params,
                                                       Mt_batched_flag)
        Gt_params_t = rejoin_batched_and_static_params(Gt_params_t, static_Gt_params,
                                                       Gt_batched_flag)

        nut_log_weight = nut_log_potential(x_t, nut_params_t)
        Mt_log_weight = Mt_log_potential(x_t_1, x_t, Mt_params_t)
        Gt_log_weight = Gt_log_potential(x_t_1, x_t, Gt_params_t)

        return Gt_log_weight + Mt_log_weight - nut_log_weight

    return log_weight_function, params


@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def compute_first_log_weights(particles,
                              M0_log_potential, M0_params,
                              qt_log_potential, qt_params, qt_batched_flags,
                              G0_log_potential, G0_params):
    """
    This computes the initialisation weights for the first timestep only.
    """
    initial_model_potential = jax.vmap(M0_log_potential, in_axes=[0, None])
    Gt_model_potential = jax.vmap(G0_log_potential, in_axes=[0, None])
    qt_model_potential = jax.vmap(qt_log_potential, in_axes=[0, None])

    log_weights = initial_model_potential(particles, M0_params)
    log_weights = log_weights + Gt_model_potential(particles, G0_params)

    # Get the param corresponding to the first timestep if params are barched, otherwise simply get the static param
    qt_parameters = jax.tree_map(lambda z, b: z[0] if b else z,
                                 qt_params, qt_batched_flags)

    return log_weights - qt_model_potential(particles, qt_parameters)


@partial(jax.jit, static_argnums=(1, 3, 4, 6, 7))
def compute_generic_log_weights(particles,
                                nut_log_potential, nut_params, nut_batched_flag,
                                qt_log_potential, qt_params, qt_batched_flag,
                                T):
    """
    This computes the log-weights at initialisation of the recursion.
    """
    # Compute the log weights corresponding to the weighting model first.
    # Very similar in effect to _make_log_weight_fn_and_params_inputs

    batched_nut_params, static_nut_params = split_batched_and_static_params(nut_params,
                                                                            nut_batched_flag)
    batched_qt_params, static_qt_params = split_batched_and_static_params(qt_params,
                                                                          qt_batched_flag)

    batched_nut_params = jax.tree_map(lambda z: z[1:] if z.shape[0] == T else z, batched_nut_params)
    batched_qt_params = jax.tree_map(lambda z: z[1:], batched_qt_params)

    @jax.vmap
    def log_weight_function(x_t, params_t):
        nut_params_t, qt_params_t = params_t
        nut_params_t = rejoin_batched_and_static_params(nut_params_t, static_nut_params,
                                                        nut_batched_flag)
        qt_params_t = rejoin_batched_and_static_params(qt_params_t, static_qt_params,
                                                       qt_batched_flag)

        nut_log_weight = jax.vmap(nut_log_potential, in_axes=[0, None])(x_t, nut_params_t)
        qt_log_weight = jax.vmap(qt_log_potential, in_axes=[0, None])(x_t, qt_params_t)

        return nut_log_weight - qt_log_weight

    return log_weight_function(particles, (batched_nut_params, batched_qt_params))


def get_log_weights(x_t_1, log_w_t_1,
                    x_t, log_w_t, params_t,
                    log_weight_fn):
    # House keeping to get the required inputs.

    # This nested vmap allows to define log_weight_fn more easily at the API level. This is to create a
    # (N,N) -> N^2 function while only having to care about elementwise formulas.
    # if log_weight_fn = lambda a, b: u * v, then this corresponds to np.outer.
    vmapped_log_weight_fn = vmap(vmap(log_weight_fn,
                                      in_axes=[None, 0, None], out_axes=0),
                                 in_axes=[0, None, None], out_axes=0)
    log_weight_increment = vmapped_log_weight_fn(x_t_1, x_t, params_t)  # shape = M, N

    # Take the corresponding time step and reshape to allow for adding residual weights in parallel

    log_weights = log_weight_increment + log_w_t_1[:, None] + log_w_t[None, :]
    return log_weights


def get_weights_batch(trajectories_a, log_weights_a,
                      trajectories_b, log_weights_b, params_b,
                      log_weight_fn: Callable[[PyTree, PyTree, Any], float]):
    # House keeping to get the required inputs.
    params_t = tree_map(lambda z: z[0], params_b)
    x_t_1 = tree_map(lambda z: z[-1], trajectories_a)
    x_t = tree_map(lambda z: z[0], trajectories_b)
    log_w_t_1 = log_weights_a[-1]
    log_w_t = log_weights_b[0]

    log_weights = get_log_weights(x_t_1, log_w_t_1,
                                  x_t, log_w_t, params_t,
                                  log_weight_fn)

    ell_inc = logsumexp(log_weights)
    weights = jnp.exp(log_weights - ell_inc)
    return weights, ell_inc


@partial(jax.jit, static_argnums=(2,))
def log_matvec(log_A, log_b, transpose_a=False):
    """
    Examples
    --------
    >>> import numpy as np
    >>> log_A = np.random.randn(50, 3)
    >>> log_b = np.random.randn(3)
    >>> np.max(np.abs(np.exp(log_matvec(log_A, log_b)) - np.exp(log_A) @ np.exp(log_b))) < 1e-5
    True
    """
    if transpose_a:
        log_A = log_A.T
    Amax = jnp.max(log_A)
    bmax = jnp.max(log_b)
    A = jnp.exp(log_A - Amax)
    b = jnp.exp(log_b - bmax)
    return Amax + bmax + jnp.log(A @ b)


def none_or_shift(x, shift):
    if x is None:
        return None
    if shift > 0:
        return jax.tree_map(lambda z: z[shift:], x)
    return jax.tree_map(lambda z: z[:shift], x)


def none_or_concat(x, y, position=1):
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree_map(lambda a, b: jnp.concatenate([a[None, ...], b]), y, x)
    else:
        return jax.tree_map(lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x)
