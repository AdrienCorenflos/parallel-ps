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
from jax.random import split
from jax.scipy.special import logsumexp

from parallel_ps.base import DensityModel, UnivariatePotentialModel, BivariatePotentialModel, DSMCState, \
    NullPotentialModel, ConditionalDensityModel, split_batched_and_static_params, rejoin_batched_and_static_params
from parallel_ps.core.resampling import multinomial


def conditional_smoother(T,
                         conditional_trajectory,
                         key: chex.PRNGKey,
                         Mt: ConditionalDensityModel, Gt: BivariatePotentialModel,
                         M0: DensityModel,
                         G0: UnivariatePotentialModel = NullPotentialModel(),
                         N: int = 100,
                         do_backward_pass: bool = True
                         ):
    """

    Parameters
    ----------
    T: int
        Total number of time steps for the state
    conditional_trajectory: PyTree
        A conditional trajectory to run conditional SMC
    key: PRNGKey
        the random JAX key used as an initialisation of the algorithm.
    Mt: ConditionalDensityModel
        The transition kernel.
    Gt: BivariatePotentialModel
        The potential function
    M0: DensityModel
        The initial prior distribution of the first time step state
    G0: UnivariatePotentialModel, optional
        The potential function for the first time step. If doing smoothing where observation happens at the second
        time step (predict first) then this should encode a 0 potential, which is the default behaviour.
    N: int, optional
        Number of particles for the final state. Default is 100
    do_backward_pass: bool
        Are we doing a backward sampling pass

    Returns
    -------
    ancestors, trajectory
    """
    filtering_static_argnums = 0, 2, 4, 5, 7, 8, 9, 11
    filtering_key, sampling_key = jax.random.split(key, 2)
    filtering_fun = jax.jit(_filtering, static_argnums=filtering_static_argnums)
    filtering_result = filtering_fun(T, key,
                                     Mt.sample, Mt.parameters, Mt.batched,
                                     Gt.log_potential, Gt.parameters, Gt.batched,
                                     M0.sample, G0.log_potential, G0.parameters,
                                     N, conditional_trajectory)

    if do_backward_pass:
        bs_static_argnums = 0, 3, 5, 6
        bs_fun = jax.jit(_backward_sampling, static_argnums=bs_static_argnums)
        return bs_fun(T, sampling_key, filtering_result, Mt.log_potential, Mt.parameters, Mt.batched, N)
    else:
        return _no_sampling_select(sampling_key, filtering_result, N)


def _filtering(T: int,
               key: chex.PRNGKey,
               Mt_sampler, Mt_params, Mt_batched_flag,
               Gt_log_potential, Gt_params, Gt_batched_flag,
               M0_sampler, G0_log_potential, G0_params,
               N, conditional_trajectory):
    key, init_key = split(key, 2)
    init_particles = M0_sampler(key, N)

    # If conditional trajectory is not None, then update the first simulation index of all the time steps.
    init_conditional_trajectory = jax.tree_map(lambda z: z[0], conditional_trajectory)
    conditional_trajectory = jax.tree_map(lambda z: z[1:], conditional_trajectory)
    init_particles = jax.tree_map(lambda z, y: z.at[0].set(y), init_particles, init_conditional_trajectory)

    vmapped_G0_log_potential = jax.vmap(lambda z: G0_log_potential(z, G0_params))
    first_log_weights = vmapped_G0_log_potential(init_particles)
    normalizer = logsumexp(first_log_weights)
    first_log_weights = first_log_weights - normalizer
    ell_0 = normalizer - math.log(N)

    Mt_sampler, _, Mt_params = _make_sampler_or_potential(Mt_sampler, None, Mt_params, Mt_batched_flag)
    _, Gt_log_potential, Gt_params = _make_sampler_or_potential(None, Gt_log_potential, Gt_params, Gt_batched_flag)

    def body(carry, inputs):
        _, curr_log_weights, curr_ell, curr_particles = carry
        op_key, Mt_params_t, Gt_params_t, curr_conditional = inputs

        weights = jnp.exp(curr_log_weights)

        op_sample_key, op_resample_key = jax.random.split(op_key)
        idx = _conditional_resample(op_resample_key, weights, N)
        curr_particles = jax.tree_map(lambda x: x[idx], curr_particles)

        proposed_particles = Mt_sampler(op_sample_key, curr_particles, Mt_params_t)
        proposed_particles = jax.tree_map(lambda y, z: z.at[0].set(y), curr_conditional, proposed_particles)

        curr_log_weights = Gt_log_potential(curr_particles, proposed_particles, Gt_params_t)
        curr_normalizer = logsumexp(curr_log_weights)
        ell_inc = curr_normalizer - math.log(N)
        curr_log_weights = curr_log_weights - curr_normalizer

        next_carry = idx, curr_log_weights, curr_ell + ell_inc, proposed_particles
        save = curr_log_weights, idx, proposed_particles, curr_ell
        return next_carry, save

    # Create inputs
    scan_keys = split(key, T - 1)

    init_ancestors = jnp.arange(N, dtype=int)
    init = init_ancestors, first_log_weights, ell_0, init_particles
    scan_inputs = scan_keys, Mt_params, Gt_params, conditional_trajectory
    _, (log_weights, ancestors, trajectories, ells) = jax.lax.scan(body, init, scan_inputs)

    trajectories = jax.tree_map(lambda z, y: jnp.insert(z, 0, y, axis=0), trajectories, init_particles)
    ancestors = jnp.insert(ancestors, 0, init_ancestors, axis=0)
    ells = jnp.insert(ells, 0, ell_0, axis=0)
    log_weights = jnp.insert(log_weights, 0, first_log_weights, axis=0)
    smc_state = DSMCState(trajectories, log_weights, ells, ancestors)

    return smc_state


def _normalize_and_exp(log_weights):
    log_nan_max = jnp.nanmax(log_weights)
    log_weights = log_weights - log_nan_max
    weights = jnp.exp(log_weights)
    sum_weights = jnp.nansum(weights)
    return weights / sum_weights


def _backward_sampling(T, key, filtering_state: DSMCState,
                       Mt_log_potential, Mt_params, Mt_batched_flag,
                       N):
    trajectories, log_weights, _, ancestors = filtering_state

    Mt_batched_params, Mt_static_params = split_batched_and_static_params(Mt_params,
                                                                          Mt_batched_flag)

    @partial(jax.vmap, in_axes=[0, None, None])
    def spec_Mt_log_potential(x_t_1, x_t, params_t):
        params_t = rejoin_batched_and_static_params(params_t, Mt_static_params,
                                                    Mt_batched_flag)
        return Mt_log_potential(x_t_1, x_t, params_t)

    init_key, key = jax.random.split(key, 2)
    B_T = jax.random.choice(init_key, N, p=jnp.exp(log_weights[-1]))
    X_T = jax.tree_map(lambda z: z[-1, B_T], trajectories)
    rest_particles = jax.tree_map(lambda z: z[:-1], trajectories)

    def body(carry, inputs):
        _, X_t_p_1 = carry
        op_key, log_weights_t, particles_t, Mt_params_t_p_1 = inputs
        log_Mt_weights = spec_Mt_log_potential(particles_t, X_t_p_1, Mt_params_t_p_1)

        curr_log_weights = log_weights_t + log_Mt_weights

        normalizer = logsumexp(curr_log_weights)
        weights = jnp.exp(curr_log_weights - normalizer)
        B_t = jax.random.choice(op_key, N, p=weights)
        X_t = jax.tree_map(lambda z: z[B_t], particles_t)
        return (B_t, X_t), (B_t, X_t)

    scan_keys = jax.random.split(key, T - 1)

    scan_inputs = scan_keys, log_weights[:-1], rest_particles, Mt_batched_params
    _, (ancestors, sampled_traj) = jax.lax.scan(body, (B_T, X_T), scan_inputs, reverse=True)
    sampled_traj = jax.tree_map(lambda z, y: jnp.concatenate([z, y[None, :]], 0), sampled_traj, X_T)
    ancestors = jnp.concatenate([ancestors, jnp.atleast_1d(B_T)], 0)
    return ancestors, sampled_traj


@partial(jax.jit, static_argnums=(2,))
def _no_sampling_select(sampling_key, filtering_result, N):
    trajectories, log_weights, _, ancestors = filtering_result
    init_key, key = jax.random.split(sampling_key, 2)
    B_T = jax.random.choice(init_key, N, p=jnp.exp(log_weights[-1]))
    X_T = jax.tree_map(lambda z: z[-1, B_T], trajectories)
    rest_particles = jax.tree_map(lambda z: z[: -1], trajectories)

    def body(B_t_p_1, inp):
        X_t, A_t = inp
        B_t = A_t[B_t_p_1]
        X_t = X_t[B_t]
        return B_t, (X_t, B_t)

    _, (sampled_traj, ancestors) = jax.lax.scan(body, B_T, (rest_particles, ancestors[1:]), reverse=True)

    sampled_traj = jax.tree_map(lambda z, y: jnp.concatenate([z, y[None, :]], 0), sampled_traj, X_T)
    ancestors = jnp.concatenate([ancestors, jnp.atleast_1d(B_T)], 0)
    return ancestors, sampled_traj


def _conditional_resample(key, weights, N):
    idx = multinomial(weights, key, N - 1)
    idx = jnp.insert(idx, 0, 0, axis=0)
    return idx


def _make_sampler_or_potential(sampler_fn, potential_fn, params, bathed_flag):
    batched_params, static_params = split_batched_and_static_params(params,
                                                                    bathed_flag)

    if potential_fn is not None:
        @partial(jax.vmap, in_axes=[0, 0, None])
        def batched_potential_fn(x_t_1, x_t, params_t):
            params_t = rejoin_batched_and_static_params(params_t, static_params,
                                                        bathed_flag)
            return potential_fn(x_t_1, x_t, params_t)
    else:
        batched_potential_fn = None

    if sampler_fn is not None:
        def batched_sampler_fn(key, x_t_1, params_t):
            params_t = rejoin_batched_and_static_params(params_t, static_params,
                                                        bathed_flag)
            return sampler_fn(key, x_t_1, params_t)
    else:
        batched_sampler_fn = None
    return batched_sampler_fn, batched_potential_fn, batched_params
