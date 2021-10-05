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
import jax.numpy as jnp
from jax import tree_map, jit, vmap
from jax.ops import index_update
from jax.scipy.special import logsumexp

from parallel_ps.base import PyTree, DSMCState
from parallel_ps.base import normalize
from parallel_ps.core.resampling import multinomial, RESAMPLING_SIGNATURE

_INPUTS_TYPE = Tuple[DSMCState, chex.PRNGKey, chex.ArrayTree or None]


@partial(jit, static_argnums=(2, 3, 4), donate_argnums=(0, 1))
def operator(inputs_a: _INPUTS_TYPE, inputs_b: _INPUTS_TYPE, log_weight_fn: Callable[[PyTree, PyTree, Any], float],
             resampling_method: RESAMPLING_SIGNATURE, conditional: bool = False):
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
    conditional: bool, optional
        Is this the conditional operator? Default is False.
    Returns
    -------

    """
    if conditional and resampling_method is not multinomial:
        raise NotImplementedError("Conditional dSMC with non-multinomial resampling is not implemented yet.")

    # Unpack the states
    state_a, keys_a, params_a = inputs_a
    state_b, keys_b, params_b = inputs_b
    trajectories_a, log_weights_a, ells_a, origins_a = state_a
    trajectories_b, log_weights_b, ells_b, origins_b = state_b

    # House keeping to get the required inputs.
    N = log_weights_a.shape[-1]
    params_t = tree_map(lambda z: z[0], params_b)
    key_t = keys_b[0]

    # Take the corresponding time step and reshape to allow for computing the mixing weights in parallel
    x_t_1 = tree_map(lambda z: z[-1], trajectories_a)
    x_t = tree_map(lambda z: z[0], trajectories_b)

    # This nested vmap allows to define log_weight_fn more easily at the API level. This is to create a
    # (N,N) -> N^2 function while only having to care about elementwise formulas.
    vmapped_log_weight_fn = vmap(vmap(log_weight_fn,
                                      in_axes=[0, None, None], out_axes=0),
                                 in_axes=[None, 0, None], out_axes=1)
    log_weight_increment = vmapped_log_weight_fn(x_t_1,  x_t, params_t)  # shape = M, N

    # Compute the log-likelihood increment
    ell_inc = logsumexp(log_weight_increment) - 2 * math.log(N)

    # Take the corresponding time step and reshape to allow for adding residual weights in parallel
    log_w_t_1 = log_weights_a[-1]
    log_w_t = log_weights_b[0]
    log_weights = log_weight_increment + log_w_t_1[None, :] + log_w_t[:, None]

    # Normalise
    weights = normalize(jnp.ravel(log_weights))  # shape = M * N

    idx = resampling_method(weights, key_t, N)  # shape = N
    left_idx, right_idx = jnp.unravel_index(idx, (N, N))

    # If we are using conditional dSMC, we need to make sure to preserve the first trajectory.
    if conditional:
        right_idx = index_update(right_idx, 0, 0)
        left_idx = index_update(left_idx, 0, 0)

    # Resample the trajectories
    trajectories_a = tree_map(lambda z: jnp.take(z, left_idx, axis=1), trajectories_a)
    trajectories_b = tree_map(lambda z: jnp.take(z, right_idx, axis=1), trajectories_b)

    # Keep track of the trajectories origins for analysis down the line (not used in the algo)
    origins_a = jnp.take(origins_a, left_idx, axis=1)
    origins_b = jnp.take(origins_b, right_idx, axis=1)

    # Gather the results
    keys = jnp.concatenate([keys_a, keys_b])
    params = tree_map(lambda a, b: jnp.concatenate([a, b]), params_a, params_b)
    origins = jnp.concatenate([origins_a, origins_b])
    trajectories = tree_map(lambda a, b: jnp.concatenate([a, b]), trajectories_a, trajectories_b)

    # Increment log-likelikelihood to the right
    ells = jnp.concatenate([ells_a, ells_a[-1] + ells_b + ell_inc])
    # Set the resulting log_weights to a constant.
    log_weights = 0. * jnp.concatenate([log_weights_a, log_weights_b])

    return DSMCState(trajectories, log_weights, ells, origins), keys, params
