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

import chex
import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp

from parallel_ps.base import DensityModel, UnivariatePotentialModel, BivariatePotentialModel, DSMCState, \
    NullPotentialModel, split_batched_and_static_params, rejoin_batched_and_static_params
from parallel_ps.core.resampling import multinomial
from parallel_ps.smoothing_utils import compute_first_log_weights, make_log_weight_fn_and_params_inputs, \
    log_matvec, none_or_shift, none_or_concat, get_log_weights


def smoothing(key: chex.PRNGKey, qt: DensityModel,
              Mt: BivariatePotentialModel, Gt: BivariatePotentialModel,
              M0: UnivariatePotentialModel,
              G0: UnivariatePotentialModel = NullPotentialModel(),
              N: int = 100,
              M: int = None):
    """
    Forward filtering backward sampling.

    Parameters
    ----------
    key: PRNGKey
        the random JAX key used as an initialisation of the algorithm.
    qt: DensityModel
        The proposal model.
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
        Number of particles used for the filtering pass. Default is 100
    M: int, optional
        Number of particles used for the sampling pass. Default is the same as N

    Returns
    -------
    smc_state: DSMCState
        The final state of the algorithm
    """
    # In the current state of JAX, you should not JIT a PMAP operation as this induces communication
    # over devices instead of using shared memory.
    if M is None:
        M = N
    filter_key, smoother_key = jax.random.split(key, 2)
    filter_trajectories, filter_log_weights = _forward_filtering(filter_key,
                                                                 qt.sample, qt.log_potential, qt.parameters, qt.batched,
                                                                 Mt.log_potential, Mt.parameters, Mt.batched,
                                                                 Gt.log_potential, Gt.parameters, Gt.batched,
                                                                 M0.log_potential, M0.parameters,
                                                                 G0.log_potential, G0.parameters,
                                                                 N,
                                                                 qt.T)

    sampled_traj, sampled_indices = _backward_sampling(smoother_key, filter_trajectories, filter_log_weights,
                                                       Mt.log_potential, Mt.parameters, Mt.batched, M)
    return sampled_traj, sampled_indices


@partial(jax.jit, static_argnums=(1, 2, 4, 5, 7, 8, 10, 11, 13, 15, 16))
def _forward_filtering(key: chex.PRNGKey,
                       qt_sample, qt_log_potential, qt_params, qt_batched_flag,
                       Mt_log_potential, Mt_params, Mt_batched_flag,
                       Gt_log_potential, Gt_params, Gt_batched_flag,
                       M0_log_potential, M0_params,
                       G0_log_potential, G0_params,
                       N, T):
    # Sample initial trajectories
    trajectories = qt_sample(key, N)

    # Compute the log weights for each time steps
    first_time_step = jax.tree_map(lambda z: z[0], trajectories)

    first_log_weights = compute_first_log_weights(first_time_step,
                                                  M0_log_potential, M0_params,
                                                  qt_log_potential, qt_params, qt_batched_flag,
                                                  G0_log_potential, G0_params)

    ell_0 = logsumexp(first_log_weights)
    first_log_weights = first_log_weights - ell_0
    ell_0 = ell_0 - math.log(N)
    log_weight_function, params_dict = make_log_weight_fn_and_params_inputs(
        qt_log_potential, qt_params, qt_batched_flag,
        Mt_log_potential, Mt_params, Mt_batched_flag,
        Gt_log_potential, Gt_params, Gt_batched_flag,
        T
    )

    zeros = jnp.zeros((N,))
    get_weights_fn = lambda x_a, x_b, p_b: get_log_weights(x_a, zeros, x_b, zeros, p_b, log_weight_function)
    log_weights_inc = jax.vmap(get_weights_fn)(none_or_shift(trajectories, -1), none_or_shift(trajectories, 1),
                                               none_or_shift(params_dict, 1))

    def forward_filtering_body(carry, transition_log_weights):
        filtering_log_weights, curr_ell = carry
        filtering_log_weights = log_matvec(transition_log_weights, filtering_log_weights, transpose_a=True)
        ell_inc = logsumexp(filtering_log_weights)
        filtering_log_weights = filtering_log_weights - ell_inc
        ell_inc = ell_inc - math.log(N)

        return (filtering_log_weights, curr_ell + ell_inc), filtering_log_weights

    (_, ell), log_weights = jax.lax.scan(forward_filtering_body, (first_log_weights, ell_0), log_weights_inc)

    log_weights = jnp.concatenate([jnp.expand_dims(first_log_weights, 0), log_weights])
    return trajectories, log_weights


@partial(jax.jit, static_argnums=(3, 5, 6))
def _backward_sampling(key: chex.PRNGKey,
                       particles,
                       log_weights,
                       Mt_log_potential, Mt_params, Mt_batched_flag,
                       M: int):
    batched_Mt_params, static_Mt_params = split_batched_and_static_params(Mt_params,
                                                                          Mt_batched_flag)
    T = log_weights.shape[0]
    keys = jax.random.split(key, T)

    @partial(jax.vmap, in_axes=(0, None, None))
    def vmapped_Mt_log_potential(x_t_1, x_t, Mt_params_t):
        Mt_params_t = rejoin_batched_and_static_params(Mt_params_t, static_Mt_params,
                                                       Mt_batched_flag)
        return Mt_log_potential(x_t_1, x_t, Mt_params_t)

    K_T = multinomial(jnp.exp(log_weights[-1]), keys[-1], M)  # noqa: bad type def in chex. TODO: send a PR.
    x_T = jax.tree_map(lambda x: x[-1, K_T], particles)

    @partial(jax.vmap, in_axes=(0, None))
    def backward_sampling_body(x_t, inputs):
        particles_t_1, M_t_params_t, log_weights_t_1, key = inputs
        smoothing_log_weights = log_weights_t_1 + vmapped_Mt_log_potential(particles_t_1, x_t, M_t_params_t)
        smoothing_log_weights = smoothing_log_weights - logsumexp(smoothing_log_weights)
        K_t_1 = multinomial(jnp.exp(smoothing_log_weights), key, 1)[0]
        x_t_1 = jax.tree_map(lambda x: x[K_t_1], particles_t_1)
        return x_t_1, (x_t_1, K_t_1)

    _, (sampled_particles, sampled_indices) = jax.lax.scan(backward_sampling_body,
                                                           x_T,
                                                           (none_or_shift(particles, -1),
                                                            batched_Mt_params,
                                                            log_weights[:-1],
                                                            keys[:-1]
                                                            ),
                                                           reverse=True
                                                           )

    sampled_particles = none_or_concat(sampled_particles, x_T, -1)
    sampled_indices = none_or_concat(sampled_indices, K_T, -1)

    return sampled_particles, sampled_indices

