#  MIT License
#
#  Copyright (c) 2021 Adrien Corenflos
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import math
from functools import partial
from typing import Any, Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_map

from parallel_ps.base import PyTree, DSMCState
from parallel_ps.core.resampling import multinomial, RESAMPLING_SIGNATURE
from parallel_ps.smoothing_utils import get_weights_batch

_INPUTS_TYPE = Tuple[DSMCState, chex.PRNGKey, chex.ArrayTree or None, chex.ArrayTree or None]


@partial(jax.jit, static_argnums=(2, 3, 4, 5), donate_argnums=(0, 1))
def operator(inputs_a: _INPUTS_TYPE, inputs_b: _INPUTS_TYPE, log_weight_fn: Callable[[PyTree, PyTree, Any], float],
             resampling_method: RESAMPLING_SIGNATURE, n_samples: int,
             conditional: bool = False):
    """
    Operator corresponding to the stitching operation of the dSMC algorithm.
    It implements both conditional (for conditional dSMC) and non-conditional operations.

    Parameters
    ----------
    inputs_a: tuple
        A tuple of three arguments.
        First one is the state of the partial dSMC smoother to the left of the current time step.
        Second are the jax random keys used for resampling at the time steps to the left of the current time step.
        Third are the parameters used to compute the mixing weights to the left of the current time step.
    inputs_b: tuple
        Same as `inputs_b` but to the right of the current time step
    log_weight_fn: callable
        Function that computes the un-normalised stitching N^2 weights, first argument is x_{t-1}, second is x_t.
        It will be automatically batched so only needs to be expressed elementwise
    resampling_method: `parallel_ps.core.resampling.RESAMPLING_SIGNATURE`
        Resampling method used in the algorithm.
    n_samples: int
        Number of samples in the resampling
    conditional: bool, optional
        Is this the conditional operator? Default is False.

    Returns
    -------

    """
    if conditional and resampling_method is not multinomial:
        raise NotImplementedError("Conditional dSMC with non-multinomial resampling is not implemented yet.")

    # Unpack the states
    state_a, keys_a, params_a, *_ = inputs_a
    state_b, keys_b, params_b, *_ = inputs_b
    trajectories_a, log_weights_a, ells_a, origins_a = state_a
    trajectories_b, log_weights_b, ells_b, origins_b = state_b

    weights, ell_inc = get_weights_batch(trajectories_a, log_weights_a,
                                         trajectories_b, log_weights_b, params_b,
                                         log_weight_fn)

    if conditional:
        idx = resampling_method(jnp.ravel(weights), keys_b[0], n_samples - 1)
        idx = jnp.insert(idx, 0, 0)
    else:
        idx = resampling_method(jnp.ravel(weights), keys_b[0], n_samples)  # shape = N
    l_idx, r_idx = jax.vmap(jnp.unravel_index, in_axes=[0, None])(idx, (n_samples, n_samples))

    return _gather_results(l_idx, r_idx, n_samples, ell_inc,
                           trajectories_a, origins_a, ells_a, log_weights_a, keys_a, params_a,
                           trajectories_b, origins_b, ells_b, log_weights_b, keys_b, params_b)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6), donate_argnums=(0, 1))
def lazy_operator(inputs_a: _INPUTS_TYPE, inputs_b: _INPUTS_TYPE, log_weight_fn: Callable[[PyTree, PyTree, Any], float],
                  n_samples: int):
    """
    Operator corresponding to the lazy operation of the dSMC algorithm.
    It implements both conditional (for conditional dSMC) and non-conditional operations.

    Parameters
    ----------
    inputs_a: tuple
        A tuple of three arguments.
        First one is the state of the partial dSMC smoother to the left of the current time step.
        Second are the jax random keys used for resampling at the time steps to the left of the current time step.
        Third are the parameters used to compute the mixing weights to the left of the current time step.
    inputs_b: tuple
        Same as `inputs_b` but to the right of the current time step
    log_weight_fn: callable
        Function that computes the un-normalised stitching N^2 weights, first argument is x_{t-1}, second is x_t.
        It will be automatically batched so only needs to be expressed elementwise
    n_samples: int
        Number of samples in the resampling

    Returns
    -------

    """
    # TODO: UNITTEST UNITTEST UNITTEST!
    # TODO: Conditional version (probs not for this paper).

    # Unpack the states
    state_a, keys_a, params_a, log_bounds_a = inputs_a
    state_b, keys_b, params_b, log_bounds_b = inputs_b
    trajectories_a, log_weights_a, ells_a, origins_a = state_a
    trajectories_b, log_weights_b, ells_b, origins_b = state_b

    params_t = tree_map(lambda z: z[0], params_b)
    x_t_1 = tree_map(lambda z: z[-1], trajectories_a)
    x_t = tree_map(lambda z: z[0], trajectories_b)

    # x_t_1_idx = hilbert_sort(x_t_1)
    # x_t_idx = hilbert_sort(x_t)

    # trajectories_a = tree_map(lambda z: z[:, x_t_1_idx], trajectories_a)
    # trajectories_b = tree_map(lambda z: z[:, x_t_idx], trajectories_b)
    #
    # origins_a = origins_a[:, x_t_1_idx]
    # origins_b = origins_b[:, x_t_idx]
    #
    # x_t_1 = x_t_1[x_t_1_idx]
    # x_t = x_t[x_t_idx]
    #
    # log_weights_a = log_weights_a[:, x_t_1_idx]
    # log_weights_b = log_weights_b[:, x_t_idx]

    log_w_t_1 = log_weights_a[-1]
    log_w_t = log_weights_b[0]

    l_idx, r_idx = _lazy_resampling(keys_b[0], x_t_1, x_t, log_w_t_1,
                                    log_w_t, params_t, log_weight_fn, n_samples, log_bounds_b[0])

    state, keys, params = _gather_results(l_idx, r_idx, n_samples, jnp.nan,
                                          trajectories_a, origins_a, ells_a, log_weights_a, keys_a, params_a,
                                          trajectories_b, origins_b, ells_b, log_weights_b, keys_b, params_b,
                                          zero_weights=True)

    log_bounds = jnp.concatenate([log_bounds_a, log_bounds_b])
    return state, keys, params, log_bounds


def _lazy_resampling(key, xs, ys, log_w_xs, log_w_ys, params_t, log_weight_fn, n_samples, log_weight_bound_t):
    def resample_one(op_key, m):
        def cond(carry):
            accepted, *_ = carry
            return ~accepted

        def body(carry):
            _, i, j, k, n_iter = carry

            log_w = log_weight_fn(xs[i], ys[j], params_t) + log_w_ys[j] + log_w_xs[i]
            # id_print((i, j), what="i, j")

            k, u_k, ij_k = jax.random.split(k, 3)
            log_u = jnp.log(jax.random.uniform(u_k))
            # id_print(log_u, what="log_u")
            # id_print(log_w - log_weight_bound_t, what="log_w - log_weight_bound_t")

            accept = log_u < log_w - log_weight_bound_t

            i, j = jax.lax.select(accept, jnp.array([i, j]), jax.random.randint(ij_k, (2,), 0, n_samples))

            # id_print(accept, what="accept")
            return accept, i, j, k, n_iter + 1

        _, I, J, _, total_n_iter = jax.lax.while_loop(cond, body, (False, m, m, op_key, 0))

        return I, J, total_n_iter

    keys = jax.random.split(key, n_samples)

    idx_x, idx_y, n_iter_array = jax.vmap(resample_one)(keys, np.arange(n_samples, dtype=int))

    return idx_x, idx_y


def _gather_results(left_idx, right_idx, n_samples, ell_inc,
                    trajectories_a, origins_a, ells_a, log_weights_a, keys_a, params_a,
                    trajectories_b, origins_b, ells_b, log_weights_b, keys_b, params_b,
                    zero_weights=False):
    # If we are using conditional dSMC, we need to make sure to preserve the first trajectory.

    # Resample the trajectories

    trajectories_a = tree_map(lambda z: jnp.take(z, left_idx, 1), trajectories_a)
    trajectories_b = tree_map(lambda z: jnp.take(z, right_idx, 1), trajectories_b)

    # Keep track of the trajectories origins for analysis down the line (not used in the algo)
    origins_a = jnp.take(origins_a, left_idx, 1)
    origins_b = jnp.take(origins_b, right_idx, 1)

    # Gather the results
    keys = jnp.concatenate([keys_a, keys_b])
    params = tree_map(lambda a, b: jnp.concatenate([a, b]), params_a, params_b)
    origins = jnp.concatenate([origins_a, origins_b])
    trajectories = tree_map(lambda a, b: jnp.concatenate([a, b]), trajectories_a, trajectories_b)

    # Increment log-likelikelihood to the right
    ells = jnp.concatenate([ells_a, ells_a[-1] + ells_b + ell_inc])
    # Set the resulting log_weights to a constant.
    if zero_weights:
        log_weights = jnp.concatenate([jnp.full_like(log_weights_a, 0.),
                                       jnp.full_like(log_weights_b, 0.)])
    else:
        log_weights = jnp.concatenate([jnp.full_like(log_weights_a, -math.log(n_samples)),
                                       jnp.full_like(log_weights_b, -math.log(n_samples))])

    return DSMCState(trajectories, log_weights, ells, origins), keys, params
