"""
Bearings-only smoothing experiment design
"""
# IMPORTS
import time
from functools import partial

import chex
import jax
import jax.numpy as jnp
import scipy.stats as stats
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.host_callback import id_tap

from examples.models.rare_events import InitialModel, TransitionKernel, PotentialKernel, InitPotentialKernel
from parallel_ps.base import PyTree, DensityModel
from parallel_ps.core.resampling import systematic
from parallel_ps.parallel_smoother import smoothing
from parallel_ps.sequential import smoothing as seq_smoothing

# CONFIG

DO_RUN = True

# PS config
backend = "gpu"
N = 250  # Number of particles
B = 1_000  # number of PSs for stats

use_sequential = False  # use the sequential algorithm instead of the parallel one.

# Model config
T, D = 1_000, 2
m0 = jnp.zeros((D,))
chol0 = jnp.eye(D)
sv = 0.2
phi = lambda x: x

# Other config
jax_seed = 42
jax_key = jax.random.PRNGKey(jax_seed)


def integrand(trajs):
    # test function for the Fisher score
    res = D * jnp.log(sv)
    squared_diff_x = jnp.sum((trajs[1:] - phi(trajs[:-1])) ** 2, -1)
    res = squared_diff_x / sv ** 3 + res
    return jnp.sum(res, 0).mean()


class QtModel(DensityModel):
    def __init__(self):
        super(QtModel, self).__init__(jnp.nan, False, T=T)

    def log_potential(self, particles: chex.ArrayTree, _: PyTree) -> jnp.ndarray:
        dist = tfp.distributions.TruncatedNormal(0., 1., -1., 1.)
        return dist.log_prob(particles).sum(-1)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        dist = tfp.distributions.TruncatedNormal(0., 1., -1., 1.)
        return dist.sample((self.T, N, D), seed=key)


@partial(jax.jit, backend=backend)
def smc(key):
    potential_fn = lambda z: jnp.log(jnp.sum(z ** 2) < 1)
    gen_observation_potential = PotentialKernel(potential_fn)
    init_observation_potential = InitPotentialKernel(potential_fn)

    transition_kernel = TransitionKernel(phi, sv * jnp.eye(D))
    initial_model = InitialModel(m0, chol0)

    if not use_sequential:
        qt = QtModel()
        result = smoothing(key, qt, qt, transition_kernel, gen_observation_potential,
                           init_observation_potential, initial_model, N=N, resampling_method=systematic)
        return result.ells[-1], (result.trajectories, result.origins)
    else:
        (ancestors, trajectories), ells = seq_smoothing(T, key, transition_kernel, gen_observation_potential,
                                                        initial_model, init_observation_potential, N, M=N,
                                                        do_backward_pass=True)
        return ells[-1], (trajectories, ancestors)


def run_experiment():
    _ell, _ = smc(jax_key)
    _ell.block_until_ready()

    @partial(jax.jit, backend=backend)
    def loop_body(i, op_key):
        def _print_fun(j, _):
            # Handmade progress bar function
            if j + 1 == B:
                print(f"\rIteration {j + 1}/{B}", end="\n", flush=False)
            else:
                print(f"\rIteration {j + 1}/{B}", end="", flush=True)

        ell, (trajs, _) = smc(op_key)
        return id_tap(_print_fun, i, result=i + 1), (integrand(trajs), ell)

    @partial(jax.jit, backend=backend)
    def get_res():
        keys = jax.random.split(jax_key, B)
        _, (test_res, ells) = jax.lax.scan(loop_body, 0, keys)
        return test_res, ells

    tic = time.time()
    test_res, ells = get_res()
    ells.block_until_ready()
    toc = time.time()
    print()
    print("Average run time: {:.3f}".format((toc - tic) / B))
    print("ELL stats", stats.describe(ells))

    print("Fisher stats", stats.describe(test_res))


if DO_RUN:
    run_experiment()
