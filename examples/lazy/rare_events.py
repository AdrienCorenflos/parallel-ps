"""
Bearings-only smoothing experiment design
"""
# IMPORTS
import math
import os
import time
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from jax.experimental.host_callback import id_tap

from examples.models.rare_events import InitialModel, TransitionKernel, PotentialKernel, InitPotentialKernel
from parallel_ps.base import PyTree, DensityModel
from parallel_ps.core.resampling import systematic
from parallel_ps.parallel_smoother import smoothing
from parallel_ps.sequential import smoothing as seq_smoothing

# CONFIG

# jax.config.update("jax_enable_x64", False)  # We need int64 for the Hilbert sorting.
DO_RUN = True

# PS config
backend = "gpu"
B = 25  # number of PSs for stats

use_sequential = False  # use the sequential algorithm instead of the parallel one.
DO_LEARN = True
LAZY = True

# Model config
D = 1
m0 = jnp.zeros((D,))
chol0 = jnp.eye(D)
phi = lambda x: x

SVS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
TS = [32,  64, 128, 256, 512]
NS = [25, 50, 100, 250, 500, 1_000, 2_500, 5_000]
# Other config

jax_seed = 42
jax_key = jax.random.PRNGKey(jax_seed)


def integrand(trajs, sv):
    # test function for the Fisher score
    res = D * jnp.log(sv)
    squared_diff_x = jnp.sum((trajs[1:] - phi(trajs[:-1])) ** 2, -1)
    res = squared_diff_x / sv ** 3 + res
    return jnp.sum(res, 0).mean()


class QtModel(DensityModel):
    def __init__(self, T):
        super(QtModel, self).__init__(0, False, T=T)

    def log_potential(self, particles: chex.ArrayTree, param: PyTree) -> chex.ArrayTree:
        return jnp.log(jnp.sum(particles ** 2) < 1)  # noqa

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        # using Muller
        norm_key, radius_key = jax.random.split(key)
        eps = jax.random.normal(key, (self.T, N, D))
        norm = jnp.linalg.norm(eps, axis=-1, keepdims=True)
        radius = jax.random.uniform(radius_key, (self.T, N, 1)) ** (1 / D)
        return radius * eps / norm


@partial(jax.jit, backend=backend, static_argnums=(3, 4))
def smc(key, bound, sv, T, n_particles):
    potential_fn = lambda z: jnp.log(jnp.sum(z ** 2) < 1)  # The potential and the uniform cancel out
    gen_observation_potential = PotentialKernel(potential_fn)
    init_observation_potential = InitPotentialKernel(potential_fn)

    transition_kernel = TransitionKernel(phi, sv * jnp.eye(D))
    initial_model = InitialModel(m0, chol0)

    if not use_sequential:
        qt = QtModel(T)
        result = smoothing(key, qt, qt, transition_kernel, gen_observation_potential,
                           init_observation_potential, initial_model, N=n_particles, resampling_method=systematic,
                           lazy=LAZY, log_weights_bounds=bound)
        return result.ells[-1], (result.trajectories, result.origins)
    else:
        (ancestors, trajectories), ells = seq_smoothing(T, key, transition_kernel, gen_observation_potential,
                                                        initial_model, init_observation_potential, n_particles,
                                                        M=n_particles,
                                                        do_backward_pass=True)
        return ells[-1], (trajectories, ancestors)


@partial(jax.jit, backend=backend, static_argnums=(4, 5))
def loop_body(i, op_key, sv, bound, T, n_particles):
    def _print_fun(j, _):
        # Handmade progress bar function
        if j + 1 == B:
            print(f"\rIteration {j + 1}/{B}", end="\n", flush=False)
        else:
            print(f"\rIteration {j + 1}/{B}", end="", flush=True)

    ell, (trajs, _) = smc(op_key, bound, sv, T, n_particles)
    return i+1, (integrand(trajs, sv), ell)


@partial(jax.jit, backend=backend, static_argnums=(2, 3))
def get_res(sv, bound, T, n_particles):
    keys = jax.random.split(jax_key, B)
    _, (test_res, ells) = jax.lax.scan(partial(loop_body, sv=sv, bound=bound, T=T, n_particles=n_particles), 0, keys)
    return test_res, ells


def run_experiment():
    runtimes = np.empty((len(SVS), len(NS), len(TS)))
    means = np.empty_like(runtimes)
    variances = np.empty_like(runtimes)

    indices = np.recarray(variances.shape,
                          dtype=[("T", int), ("N", int), ("SV", float)])

    for j, N in enumerate(tqdm.tqdm(NS)):
        indices[:, j, :]["N"] = N
        for k, T in enumerate(tqdm.tqdm(TS)):
            indices[:, j, k]["T"] = T
            indices[:, j, k]["SV"] = SVS
            # compilation loop with very high variance to go fast.
            log_weights_bound = -D * (math.log(0.5) + 50 * math.log(2 * math.pi)) * np.ones((T - 1,))
            log_weights_bound = np.insert(log_weights_bound, 0, -D * 0.5 * math.log(2 * math.pi))
            try:
                test_res, ells = get_res(50, log_weights_bound, T, N)
                ells.block_until_ready()
            except:  # noqa: I don't care what the error is, it's a memory issue anyway.
                runtimes[:, j, k] = np.inf
                means[:, j, k] = np.nan
                means[:, j, k] = np.nan
                continue

            for i, sv in enumerate(tqdm.tqdm(SVS)):

                log_weights_bound = -D * (math.log(sv) + 0.5 * math.log(2 * math.pi)) * np.ones((T - 1,))
                log_weights_bound = np.insert(log_weights_bound, 0, -D * 0.5 * math.log(2 * math.pi))

                tic = time.time()
                test_res, ells = get_res(sv, log_weights_bound, T, N)
                ells.block_until_ready()
                toc = time.time()
                means[i, j, k] = test_res.mean()
                variances[i, j, k] = test_res.var()
                runtimes[i, j, k] = (toc - tic) / B
    return indices, means, variances, runtimes


res_indices, res_means, res_variances, res_runtimes = run_experiment()

os.makedirs("./output", exist_ok=True)
np.savez(f"./output/rare-events-{use_sequential}-{LAZY}",
         indices=res_indices,
         means=res_means,
         variances=res_variances,
         runtimes=res_runtimes)
