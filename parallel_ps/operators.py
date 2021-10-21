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
import jax.ops as ops
from jax import tree_map

from parallel_ps.base import PyTree, DSMCState
from parallel_ps.core.resampling import multinomial, RESAMPLING_SIGNATURE
from parallel_ps.smoothing_utils import get_weights_batch

_INPUTS_TYPE = Tuple[DSMCState, chex.PRNGKey, chex.ArrayTree or None]


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
    state_a, keys_a, params_a = inputs_a
    state_b, keys_b, params_b = inputs_b
    trajectories_a, log_weights_a, ells_a, origins_a = state_a
    trajectories_b, log_weights_b, ells_b, origins_b = state_b

    weights, ell_inc = get_weights_batch(trajectories_a, log_weights_a,
                                         trajectories_b, log_weights_b, params_b,
                                         log_weight_fn)

    idx = resampling_method(jnp.ravel(weights), keys_b[0], n_samples)  # shape = N
    l_idx, r_idx = jax.vmap(jnp.unravel_index, in_axes=[0, None])(idx, (n_samples, n_samples))

    return _gather_results(l_idx, r_idx, n_samples, conditional, ell_inc,
                           trajectories_a, origins_a, ells_a, log_weights_a, keys_a, params_a,
                           trajectories_b, origins_b, ells_b, log_weights_b, keys_b, params_b)


def _gather_results(left_idx, right_idx, n_samples, conditional, ell_inc,
                    trajectories_a, origins_a, ells_a, log_weights_a, keys_a, params_a,
                    trajectories_b, origins_b, ells_b, log_weights_b, keys_b, params_b):
    # If we are using conditional dSMC, we need to make sure to preserve the first trajectory.
    if conditional:
        right_idx = ops.index_update(right_idx, 0, 0)
        left_idx = ops.index_update(left_idx, 0, 0)

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
    log_weights = jnp.concatenate([jnp.full_like(log_weights_a, -math.log(n_samples)),
                                   jnp.full_like(log_weights_b, -math.log(n_samples))])

    return DSMCState(trajectories, log_weights, ells, origins), keys, params