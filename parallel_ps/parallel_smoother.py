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

from functools import partial
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import jax.ops as ops
from jax.random import split
from jax.scipy.special import logsumexp

from parallel_ps.base import DensityModel, UnivariatePotentialModel, BivariatePotentialModel, PyTree, DSMCState, \
    NullPotentialModel
from parallel_ps.core import dc_map
from parallel_ps.core.resampling import RESAMPLING_SIGNATURE, multinomial
from parallel_ps.operators import operator
from parallel_ps.smoothing_utils import make_log_weight_fn_and_params_inputs, compute_first_log_weights, \
    compute_generic_log_weights


def smoothing(key: chex.PRNGKey, qt: DensityModel, nut: UnivariatePotentialModel,
              Mt: BivariatePotentialModel, Gt: BivariatePotentialModel,
              M0: UnivariatePotentialModel,
              G0: UnivariatePotentialModel = NullPotentialModel(),
              resampling_method: RESAMPLING_SIGNATURE = multinomial,
              N: int = 100, conditional_trajectory: Optional[PyTree] = None):
    """

    Parameters
    ----------
    key: PRNGKey
        the random JAX key used as an initialisation of the algorithm.
    qt: DensityModel
        The proposal model.
    nut: DensityModel
        The independent weighting function used in the stitching weight definition.
    Mt: BivariatePotentialModel
        The transition kernel.
    Gt: BivariatePotentialModel
        The potential function
    M0: UnivariatePotentialModel
        The initial prior distribution of the first time step state
    G0: UnivariatePotentialModel, optional
        The potential function for the first time step. If doing smoothing where observation happens at the second
        time step (predict first) then this should encode a 0 potential, which is the default behaviour.
    resampling_method: callable
        Resampling method used
    N: int, optional
        Number of particles for the final state. Default is 100
    conditional_trajectory: PyTree, optional
        A conditional trajectory. If None (default) is passed, then the algorithm will be a standard particle smoother.

    Returns
    -------
    smc_state: DSMCState
        The final state of the algorithm
    """
    # In the current state of JAX, you should not JIT a PMAP operation as this induces communication
    # over devices instead of using shared memory.
    static_argnums = 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19, 21, 22, 23
    smoothing_fun = jax.jit(_smoothing, static_argnums=static_argnums)
    return smoothing_fun(key,
                         qt.sample, qt.log_potential, qt.parameters, qt.batched,
                         nut.log_potential, nut.parameters, nut.batched,
                         Mt.log_potential, Mt.parameters, Mt.batched,
                         Gt.log_potential, Gt.parameters, Gt.batched,
                         M0.log_potential, M0.parameters,
                         G0.log_potential, G0.parameters,
                         resampling_method,
                         N,
                         conditional_trajectory,
                         qt.T)


def _smoothing(key: chex.PRNGKey,
               qt_sample, qt_log_potential, qt_params, qt_batched_flag,
               nut_log_potential, nut_params, nut_batched_flag,
               Mt_log_potential, Mt_params, Mt_batched_flag,
               Gt_log_potential, Gt_params, Gt_batched_flag,
               M0_log_potential, M0_params,
               G0_log_potential, G0_params,
               resampling_method: RESAMPLING_SIGNATURE,
               N,
               conditional_trajectory,
               T):
    key, init_key = split(key, 2)

    # Sample initial trajectories
    trajectories = qt_sample(init_key, N)

    # If conditional trajectory is not None, then update the first simulation index of all the time steps.
    if conditional_trajectory is not None:
        trajectories = jax.tree_map(lambda xs, x: ops.index_update(xs, ops.index[:, 0], x),
                                    trajectories,
                                    conditional_trajectory)

    # Compute the log weights for each time steps
    first_time_step = jax.tree_map(lambda z: z[0], trajectories)
    rest_time_steps = jax.tree_map(lambda z: z[1:], trajectories)

    first_log_weights = compute_first_log_weights(first_time_step,
                                                  M0_log_potential, M0_params,
                                                  qt_log_potential, qt_params, qt_batched_flag,
                                                  G0_log_potential, G0_params)
    rest_log_weights = compute_generic_log_weights(rest_time_steps,
                                                   nut_log_potential, nut_params, nut_batched_flag,
                                                   qt_log_potential, qt_params, qt_batched_flag,
                                                   T)
    log_weights = jnp.concatenate([jnp.expand_dims(first_log_weights, 0), rest_log_weights])
    # Compute the initial log-likelihood as a log mean exp operation.

    logsumexp_weights = logsumexp(log_weights, axis=1)

    log_weights = log_weights - logsumexp_weights[:, None]  # normalize

    ells = jnp.zeros((T,))
    # Get the log_weights and required batched input to it.
    log_weight_function, params_dict = make_log_weight_fn_and_params_inputs(
        nut_log_potential, nut_params, nut_batched_flag,
        Mt_log_potential, Mt_params, Mt_batched_flag,
        Gt_log_potential, Gt_params, Gt_batched_flag,
        T
    )

    # Create inputs
    combination_keys = split(key, T)
    origins = jnp.repeat(jnp.arange(0, N)[None, :], T, axis=0)
    dsmc_state = DSMCState(trajectories, log_weights, ells, origins)
    inputs = dsmc_state, combination_keys, params_dict

    combination_operator: Callable = jax.vmap(partial(operator,
                                                      log_weight_fn=log_weight_function,
                                                      resampling_method=resampling_method,
                                                      conditional=conditional_trajectory is not None,
                                                      n_samples=N))

    final_states, *_ = dc_map(inputs, combination_operator)
    if conditional_trajectory is not None:
        final_trajectories = jax.tree_map(lambda z: z[:, 1:], final_states.trajectories)
        final_log_weights = final_states.log_weights[:, 1:]
        final_origins = final_states.origins[:, 1:]
        final_states = DSMCState(final_trajectories, final_log_weights, ells, final_origins)
    return final_states


