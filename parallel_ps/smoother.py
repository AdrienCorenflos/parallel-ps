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
from jax.random import split
from jax.scipy.special import logsumexp

from parallel_ps.base import DensityModel, UnivariatePotentialModel, BivariatePotentialModel, PyTree, DSMCState, \
    split_batched_and_static_params, rejoin_batched_and_static_params
from parallel_ps.core import dc_vmap
from parallel_ps.core.resampling import RESAMPLING_SIGNATURE
from parallel_ps.operator import operator


def smoothing(key: chex.PRNGKey, proposal_model: DensityModel, weighting_model: UnivariatePotentialModel,
              transition_model: BivariatePotentialModel, potential_model: BivariatePotentialModel,
              initial_model: UnivariatePotentialModel, initial_potential_model: UnivariatePotentialModel,
              resampling_method: RESAMPLING_SIGNATURE,
              N: int = 100, conditional_trajectory: Optional[PyTree] = None,
              dc_map: callable = dc_vmap):
    key, init_key = split(key, 2)

    # Sample initial trajectories
    trajectories = proposal_model.sample(init_key, N)

    # Get the total number of time steps. Bit complicated, but cheep, so...
    T = proposal_model.T

    # If conditional trajectory is not None, then update the first simulation index of all the time steps.
    if conditional_trajectory is not None:
        trajectories = jax.tree_map(lambda xs, x: ops.index_update(xs, ops.index[:, 0], x),
                                    trajectories,
                                    conditional_trajectory)

    # Compute the log weights for each time steps
    first_time_step = jax.tree_map(lambda z: z[0], trajectories)
    rest_time_steps = jax.tree_map(lambda z: z[1:], trajectories)

    first_log_weights = _compute_first_log_weights(first_time_step, initial_model, proposal_model,
                                                   initial_potential_model)
    rest_log_weights = _compute_generic_log_weights(rest_time_steps, weighting_model, proposal_model, T)
    log_weights = jnp.concatenate([jnp.expand_dims(first_log_weights, 0), rest_log_weights])
    # Compute the initial log-likelihood as a log mean exp operation.
    ells = logsumexp(log_weights, 1) - math.log(N)

    # Get the log_weights and required batched input to it.
    log_weight_function, params_dict = _make_log_weight_fn_and_params_inputs(weighting_model,
                                                                             transition_model,
                                                                             potential_model,
                                                                             T)

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
    return final_states


def _make_log_weight_fn_and_params_inputs(weighting_model: UnivariatePotentialModel,
                                          transition_model: BivariatePotentialModel,
                                          potential_model: BivariatePotentialModel,
                                          T: int):
    # The way we build this function is so that if the parameters are static for the model we don't duplicate them to
    # be passed to the divide and conquer map.

    batched_weighting_params, static_weighting_params = split_batched_and_static_params(weighting_model.parameters,
                                                                                        weighting_model.batched)
    batched_transition_params, static_transition_params = split_batched_and_static_params(transition_model.parameters,
                                                                                          transition_model.batched)
    batched_potential_params, static_potential_params = split_batched_and_static_params(potential_model.parameters,
                                                                                        potential_model.batched)

    # We pad as the first index is not used but the shapes need to be consistent.
    def _pad_params(batch_param):
        ndim = batch_param.ndim
        t = batch_param.shape[0]
        return jnp.pad(batch_param, [(T - t, 0)] + [(0, 0)] * (ndim - 1), constant_values=0)

    batched_weighting_params = jax.tree_map(_pad_params, batched_weighting_params)
    batched_transition_params = jax.tree_map(_pad_params, batched_transition_params)
    batched_potential_params = jax.tree_map(_pad_params, batched_potential_params)

    @jax.jit
    def log_weight_function(x_t_1, x_t, params_t):
        weighting_params_t, transition_params_t, potential_params_t = params_t
        weighting_params_t = rejoin_batched_and_static_params(weighting_params_t, static_weighting_params,
                                                              weighting_model.batched)
        transition_params_t = rejoin_batched_and_static_params(transition_params_t, static_transition_params,
                                                               transition_model.batched)
        potential_params_t = rejoin_batched_and_static_params(potential_params_t, static_potential_params,
                                                              potential_model.batched)

        weighting_log_weight = weighting_model.log_potential(x_t, weighting_params_t)
        transition_log_weight = transition_model.log_potential(x_t_1, x_t, transition_params_t)
        potential_log_weight = potential_model.log_potential(x_t_1, x_t, potential_params_t)

        return potential_log_weight + transition_log_weight - weighting_log_weight

    params = batched_weighting_params, batched_transition_params, batched_potential_params

    return log_weight_function, params


def _compute_first_log_weights(particles, initial_model, proposal_model, initial_potential_model):
    initial_model_potential = jax.vmap(initial_model.log_potential, in_axes=[0, None])
    potential_model_potential = jax.vmap(initial_potential_model.log_potential, in_axes=[0, None])
    proposal_model_potential = jax.vmap(proposal_model.log_potential, in_axes=[0, None])

    log_weights = initial_model_potential(particles, initial_model.parameters)
    log_weights = log_weights + potential_model_potential(particles, initial_potential_model.parameters)

    proposal_parameters = jax.tree_map(lambda z, b: z[0] if b else z,
                                       proposal_model.parameters,
                                       proposal_model.batched)

    return log_weights - proposal_model_potential(particles, proposal_parameters)


def _compute_generic_log_weights(particles, weighting_model, proposal_model, T):
    # Compute the log weights corresponding to the weighting model first.
    # Very similar in effect to _make_log_weight_fn_and_params_inputs

    batched_weighting_params, static_weighting_params = split_batched_and_static_params(weighting_model.parameters,
                                                                                        weighting_model.batched)
    batched_proposal_params, static_proposal_params = split_batched_and_static_params(proposal_model.parameters,
                                                                                      proposal_model.batched)

    batched_weighting_params = jax.tree_map(lambda z: z[1:] if z.shape[0] == T else z, batched_weighting_params)
    batched_proposal_params = jax.tree_map(lambda z: z[1:], batched_proposal_params)

    @jax.vmap
    def log_weight_function(x_t, params_t):
        weighting_params_t, proposal_params_t = params_t
        weighting_params_t = rejoin_batched_and_static_params(weighting_params_t, static_weighting_params,
                                                              weighting_model.batched)
        proposal_params_t = rejoin_batched_and_static_params(proposal_params_t, static_proposal_params,
                                                             proposal_model.batched)

        weighting_log_weight = jax.vmap(weighting_model.log_potential, in_axes=[0, None])(x_t, weighting_params_t)
        proposal_log_weight = jax.vmap(proposal_model.log_potential, in_axes=[0, None])(x_t, proposal_params_t)

        return weighting_log_weight - proposal_log_weight

    return log_weight_function(particles, (batched_weighting_params, batched_proposal_params))
