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
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import jax.ops as ops
from jax.experimental.host_callback import id_print
from jax.random import split
from jax.scipy.special import logsumexp

from parallel_ps.base import DensityModel, UnivariatePotentialModel, BivariatePotentialModel, PyTree, DSMCState, \
    split_batched_and_static_params, rejoin_batched_and_static_params, NullPotentialModel, normalize
from parallel_ps.core import dc_vmap
from parallel_ps.core.resampling import RESAMPLING_SIGNATURE, multinomial
from parallel_ps.operator import operator


def smoothing(key: chex.PRNGKey, qt: DensityModel, nut: UnivariatePotentialModel,
              Mt: BivariatePotentialModel, Gt: BivariatePotentialModel,
              M0: UnivariatePotentialModel,
              G0: UnivariatePotentialModel = NullPotentialModel(),
              resampling_method: RESAMPLING_SIGNATURE = multinomial,
              N: int = 100, conditional_trajectory: Optional[PyTree] = None,
              dc_map: callable = dc_vmap):
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
    dc_map: callable
        Divide and conquer utility used for the parallelisation. One of dc_vmap, or dc_pmap specialised to given
        devices.

    Returns
    -------
    smc_state: DSMCState
        The final state of the algorithm
    """
    # In the current state of JAX, you should not JIT a PMAP operation as this induces communication
    # over devices instead of using shared memory.
    if dc_map is dc_vmap:
        static_argnums = 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19, 21, 22
        smoothing_fun = jax.jit(_smoothing, static_argnums=static_argnums)
    else:
        smoothing_fun = _smoothing

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
                         dc_map,
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
               dc_map,
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

    first_log_weights = _compute_first_log_weights(first_time_step,
                                                   M0_log_potential, M0_params,
                                                   qt_log_potential, qt_params, qt_batched_flag,
                                                   G0_log_potential, G0_params)
    rest_log_weights = _compute_generic_log_weights(rest_time_steps,
                                                    nut_log_potential, nut_params, nut_batched_flag,
                                                    qt_log_potential, qt_params, qt_batched_flag,
                                                    T)
    log_weights = jnp.concatenate([jnp.expand_dims(first_log_weights, 0), rest_log_weights])
    # Compute the initial log-likelihood as a log mean exp operation.

    logsumexp_weights = logsumexp(log_weights, axis=1)

    log_weights = log_weights - logsumexp_weights[:, None]  # normalize

    ells = logsumexp_weights - math.log(N)

    # Get the log_weights and required batched input to it.
    log_weight_function, params_dict = _make_log_weight_fn_and_params_inputs(
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

    # Get the right operator
    combination_operator: Callable = partial(operator,
                                             log_weight_fn=log_weight_function,
                                             resampling_method=resampling_method,
                                             conditional=conditional_trajectory is not None)

    final_states, *_ = dc_map(inputs, combination_operator)
    if conditional_trajectory is not None:
        final_trajectories = jax.tree_map(lambda z: z[:, 1:], final_states.trajectories)
        final_log_weights = final_states.log_weights[:, 1:]
        final_origins = final_states.origins[:, 1:]
        final_states = DSMCState(final_trajectories, final_log_weights, ells, final_origins)
    return final_states


def _make_log_weight_fn_and_params_inputs(nut_log_potential, nut_params, nut_batched_flag,
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

    params = batched_nut_params, batched_Mt_params, batched_Gt_params

    return log_weight_function, params


@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def _compute_first_log_weights(particles,
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
                                 qt_params,
                                 qt_batched_flags)

    return log_weights - qt_model_potential(particles, qt_parameters)


@partial(jax.jit, static_argnums=(1, 3, 4, 6, 7))
def _compute_generic_log_weights(particles,
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
