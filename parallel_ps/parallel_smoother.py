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
import warnings
from functools import partial
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
from jax.random import split
from jax.scipy.special import logsumexp

from parallel_ps.base import DensityModel, UnivariatePotentialModel, BivariatePotentialModel, PyTree, DSMCState, \
    NullPotentialModel
from parallel_ps.core import dc_map
from parallel_ps.core.resampling import RESAMPLING_SIGNATURE, multinomial
from parallel_ps.operators import operator, lazy_operator
from parallel_ps.smoothing_utils import make_log_weight_fn_and_params_inputs, compute_first_log_weights, \
    compute_generic_log_weights, get_log_weights


def smoothing(key: chex.PRNGKey, qt: DensityModel, nut: UnivariatePotentialModel,
              Mt: BivariatePotentialModel, Gt: BivariatePotentialModel,
              M0: UnivariatePotentialModel,
              G0: UnivariatePotentialModel = NullPotentialModel(),
              resampling_method: RESAMPLING_SIGNATURE = multinomial,
              N: int = 100, conditional_trajectory: Optional[PyTree] = None,
              lazy: bool = False, log_weights_bounds: chex.Array = None):
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
    lazy: bool, optional
        Use the lazy operator based on rejection resampling. Default is False. If True, then log_weights_bounds must be
        passed for all time steps > 1. For the time being we only support conditional_trajectory=None, qt = nut.
        Anything else will result in an error (hopefully) or statistically wrong results (if not caught).
    log_weights_bounds: array_like
        The upper bound for all the weights.

    Returns
    -------
    smc_state: DSMCState
        The final state of the algorithm
    """

    if lazy:
        if any([conditional_trajectory is not None, (qt is not nut) and (nut is not None), log_weights_bounds is None]):
            raise NotImplementedError("See docstring for `lazy`")
        static_argnums = 1, 2, 4, 5, 7, 8, 10, 11, 13, 16, 17
        smoothing_fun = jax.jit(_lazy_smoothing, static_argnums=static_argnums)
        return smoothing_fun(key,
                             qt.sample, qt.log_potential, qt.parameters, qt.batched,
                             Mt.log_potential, Mt.parameters, Mt.batched,
                             Gt.log_potential, Gt.parameters, Gt.batched,
                             M0.log_potential, M0.parameters,
                             G0.log_potential, G0.parameters,
                             log_weights_bounds, N, qt.T)

    static_argnums = 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19, 21, 22
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

    log_weights, log_weight_function, params_dict, trajectories = _init(init_key,
                                                                        qt_sample, qt_log_potential, qt_params,
                                                                        qt_batched_flag,
                                                                        nut_log_potential, nut_params, nut_batched_flag,
                                                                        Mt_log_potential, Mt_params, Mt_batched_flag,
                                                                        Gt_log_potential, Gt_params, Gt_batched_flag,
                                                                        M0_log_potential, M0_params,
                                                                        G0_log_potential, G0_params,
                                                                        N, T, conditional_trajectory)

    # Compute the initial log-likelihood as a log mean exp operation.
    logsumexp_weights = logsumexp(log_weights, axis=1)
    log_weights = log_weights - logsumexp_weights[:, None]  # normalize
    ells = jnp.zeros((T,))

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
    return final_states


def _lazy_smoothing(key: chex.PRNGKey,
                    qt_sample, qt_log_potential, qt_params, qt_batched_flag,
                    Mt_log_potential, Mt_params, Mt_batched_flag,
                    Gt_log_potential, Gt_params, Gt_batched_flag,
                    M0_log_potential, M0_params,
                    G0_log_potential, G0_params,
                    log_weights_bounds,
                    N, T):
    key, init_key = split(key, 2)

    log_weight_function, params_dict, trajectories, log_weights = _lazy_init(init_key,
                                                                             qt_sample, qt_log_potential, qt_params,
                                                                             qt_batched_flag,
                                                                             Mt_log_potential, Mt_params,
                                                                             Mt_batched_flag,
                                                                             Gt_log_potential, Gt_params,
                                                                             Gt_batched_flag,
                                                                             M0_log_potential, M0_params,
                                                                             G0_log_potential, G0_params,
                                                                             N, T)

    log_weights_bounds = log_weights_bounds.at[1].add(log_weights_bounds[0])
    # All the initial weights are assumed to be 1 in this algorithm
    ells = jnp.nan * jnp.ones((T,))

    # Create inputs
    combination_keys = split(key, T)
    origins = jnp.repeat(jnp.arange(0, N)[None, :], T, axis=0)
    dsmc_state = DSMCState(trajectories, log_weights, ells, origins)

    if len(log_weights_bounds) < T:
        raise ValueError("Bounds need to be explicitly given for each time step")

    inputs = dsmc_state, combination_keys, params_dict, log_weights_bounds

    combination_operator: Callable = jax.vmap(partial(lazy_operator,
                                                      log_weight_fn=log_weight_function,
                                                      n_samples=N))

    final_states, *_ = dc_map(inputs, combination_operator)
    return final_states


def loss_fn(key: chex.PRNGKey, qt: DensityModel, nut: UnivariatePotentialModel,
            Mt: BivariatePotentialModel, Gt: BivariatePotentialModel,
            M0: UnivariatePotentialModel,
            G0: UnivariatePotentialModel = NullPotentialModel(),
            N: int = 100):
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
    N: int, optional
        Number of particles for the final state. Default is 100

    Returns
    -------
    loss: float
        The The loss function
    """

    static_argnums = 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19
    _loss_fun = jax.jit(_loss, static_argnums=static_argnums)
    return _loss_fun(key,
                     qt.sample, qt.log_potential, qt.parameters, qt.batched,
                     nut.log_potential, nut.parameters, nut.batched,
                     Mt.log_potential, Mt.parameters, Mt.batched,
                     Gt.log_potential, Gt.parameters, Gt.batched,
                     M0.log_potential, M0.parameters,
                     G0.log_potential, G0.parameters,
                     N,
                     qt.T)


def _loss(key: chex.PRNGKey,
          qt_sample, qt_log_potential, qt_params, qt_batched_flag,
          nut_log_potential, nut_params, nut_batched_flag,
          Mt_log_potential, Mt_params, Mt_batched_flag,
          Gt_log_potential, Gt_params, Gt_batched_flag,
          M0_log_potential, M0_params,
          G0_log_potential, G0_params,
          N, T):
    log_weights, log_weight_function, params_dict, trajectories = _init(key,
                                                                        qt_sample, qt_log_potential, qt_params,
                                                                        qt_batched_flag,
                                                                        nut_log_potential, nut_params, nut_batched_flag,
                                                                        Mt_log_potential, Mt_params, Mt_batched_flag,
                                                                        Gt_log_potential, Gt_params, Gt_batched_flag,
                                                                        M0_log_potential, M0_params,
                                                                        G0_log_potential, G0_params,
                                                                        N, T, None)

    vmapped_get_log_weights = jax.vmap(get_log_weights, in_axes=[0, 0, 0, 0, 0, None])
    x_t, log_w_t, params_t = jax.tree_map(lambda z: z[1:], [trajectories, log_weights, params_dict])
    x_t_1, log_w_t_1 = jax.tree_map(lambda z: z[:-1], [trajectories, log_weights])

    log_weights = vmapped_get_log_weights(x_t_1, log_w_t_1, x_t, log_w_t, params_t, log_weight_function)
    weights = jnp.exp(log_weights)
    weights = weights.reshape(T - 1, -1)
    return jnp.mean(jnp.var(weights, axis=-1))


def _init(key: chex.PRNGKey,
          qt_sample, qt_log_potential, qt_params, qt_batched_flag,
          nut_log_potential, nut_params, nut_batched_flag,
          Mt_log_potential, Mt_params, Mt_batched_flag,
          Gt_log_potential, Gt_params, Gt_batched_flag,
          M0_log_potential, M0_params,
          G0_log_potential, G0_params,
          N, T, conditional_trajectory):
    # Sample initial trajectories
    trajectories = qt_sample(key, N)

    # If conditional trajectory is not None, then update the first simulation index of all the time steps.
    if conditional_trajectory is not None:
        trajectories = jax.tree_map(lambda xs, x: xs.at[:, 0].set(x),
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

    # Get the log_weights and required batched input to it.
    log_weight_function, params_dict = make_log_weight_fn_and_params_inputs(
        nut_log_potential, nut_params, nut_batched_flag,
        Mt_log_potential, Mt_params, Mt_batched_flag,
        Gt_log_potential, Gt_params, Gt_batched_flag,
        T
    )
    return log_weights, log_weight_function, params_dict, trajectories


def _lazy_init(key: chex.PRNGKey,
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
    rest_log_weights = jnp.zeros((T - 1, N))
    log_weights = jnp.concatenate([jnp.expand_dims(first_log_weights, 0), rest_log_weights])

    # Get the log_weights and required batched input to it.
    log_weight_function, params_dict = make_log_weight_fn_and_params_inputs(
        qt_log_potential, qt_params, qt_batched_flag,
        Mt_log_potential, Mt_params, Mt_batched_flag,
        Gt_log_potential, Gt_params, Gt_batched_flag,
        T
    )
    return log_weight_function, params_dict, trajectories, log_weights
