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

from parallel_ps.base import DensityModel, UnivariatePotentialModel, BivariatePotentialModel, PyTree, DSMCState
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
    T = jax.tree_flatten(trajectories)[0][0].shape[0]

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
    rest_log_weights = _compute_generic_log_weights(rest_time_steps, weighting_model, proposal_model)
    log_weights = jnp.concatenate([jnp.expand_dims(first_log_weights, 0), rest_log_weights])
    # Compute the initial log-likelihood as a log mean exp operation.
    ells = logsumexp(log_weights, 1) - math.log(N)

    # Get the log_weights and required batched input to it.
    log_weight_function, params_dict = _make_log_weight_fn_and_params_inputs(weighting_model,
                                                                             transition_model,
                                                                             potential_model)

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
                                          potential_model: BivariatePotentialModel):
    # The way we build this function is so that if the parameters are static for the model we don't duplicate them to
    # be passed to the divide and conquer map.

    @jax.vmap
    def log_weight_function(x_t_1, x_t, params_t):
        weighting_params = params_t.get("weighting_params", weighting_model.parameters)
        transition_params = params_t.get("transition_params", transition_model.parameters)
        potential_params = params_t.get("potential_params", potential_model.parameters)

        weighting_log_weight = weighting_model.log_potential(x_t, weighting_params)
        transition_log_weight = transition_model.log_potential(x_t_1, x_t, transition_params)
        potential_log_weight = potential_model.log_potential(x_t_1, x_t, potential_params)

        return potential_log_weight + transition_log_weight - weighting_log_weight

    params_dict = {}
    if weighting_model.batched:
        params_dict["weighting_params"] = jax.tree_map(lambda z: z[1:], weighting_model.parameters)
    if weighting_model.batched:
        params_dict["transition_params"] = jax.tree_map(lambda z: z[1:], transition_model.parameters)
    if weighting_model.batched:
        params_dict["potential_params"] = jax.tree_map(lambda z: z[1:], potential_model.parameters)

    return log_weight_function, params_dict


def _compute_first_log_weights(particles, initial_model, proposal_model, initial_potential_model):
    log_weights = initial_model.log_potential(particles, initial_model.parameters)
    log_weights = log_weights + initial_potential_model.log_potential(particles, initial_potential_model.parameters)
    if proposal_model.batched:
        proposal_parameters = jax.tree_map(lambda z: z[0], proposal_model.parameters)
    else:
        proposal_parameters = proposal_model.parameters

    return log_weights - proposal_model.log_potential(particles, proposal_parameters)


def _compute_generic_log_weights(particles, weighting_model, proposal_model):
    # Compute the log weights corresponding to the weighting model first
    if weighting_model.batched:
        weighting_parameters = jax.tree_map(lambda z: z[1:], weighting_model.parameters)
        weighting_log_weights = jax.vmap(weighting_model.log_potential)(particles, weighting_parameters)
    else:
        weighting_log_weights = jax.vmap(weighting_model.log_potential, in_axes=[0, None])(particles,
                                                                                           weighting_model.parameters)
    # Then compute the log weights corresponding to the proposal model
    if proposal_model.batched:
        proposal_parameters = jax.tree_map(lambda z: z[1:], proposal_model.parameters)
        proposal_log_weights = jax.vmap(proposal_model.log_potential)(particles, proposal_parameters)
    else:
        proposal_log_weights = jax.vmap(proposal_model.log_potential, in_axes=[0, None])(particles,
                                                                                         proposal_model.parameters)

    return weighting_log_weights - proposal_log_weights
