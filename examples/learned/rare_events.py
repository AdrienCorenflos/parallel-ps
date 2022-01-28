"""
Bearings-only smoothing experiment design
"""
# IMPORTS
import time
from functools import partial

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.stats as stats
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.host_callback import id_tap
from jax.experimental.optimizers import adam

from examples.models.rare_events import InitialModel, TransitionKernel, PotentialKernel, InitPotentialKernel
from parallel_ps.base import PyTree, DensityModel
from parallel_ps.core.resampling import systematic
from parallel_ps.parallel_smoother import smoothing, loss_fn
from parallel_ps.sequential import smoothing as seq_smoothing

# CONFIG

DO_RUN = True

# PS config
backend = "gpu"
N = 250  # Number of particles
B = 1_000  # number of PSs for stats

use_sequential = True  # use the sequential algorithm instead of the parallel one.
DO_LEARN = True

# Model config
T, D = 100, 2
m0 = jnp.zeros((D,))
chol0 = jnp.eye(D)
sv = 0.2
phi = lambda x: x

# Other config
jax_seed = 42
jax_key = jax.random.PRNGKey(jax_seed)
n_steps = 500
M = 20
learning_rate = 1e-3
optimizer = adam(learning_rate)


def integrand(trajs):
    # test function for the Fisher score
    res = D * jnp.log(sv)
    squared_diff_x = jnp.sum((trajs[1:] - phi(trajs[:-1])) ** 2, -1)
    res = squared_diff_x / sv ** 3 + res
    return jnp.sum(res, 0).mean()


class QtModel(DensityModel):
    def __init__(self, sigmas):
        super(QtModel, self).__init__(sigmas, True, T=T)

    def log_potential(self, particles: chex.ArrayTree, sigma: PyTree) -> jnp.ndarray:
        dist = tfp.distributions.TruncatedNormal(0., sigma, -1., 1.)
        return dist.log_prob(particles).sum(-1)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        dist = tfp.distributions.TruncatedNormal(0., self.parameters, -1., 1.)
        samples = dist.sample((N,), seed=key)
        return jnp.transpose(samples, [1, 0, 2])


@partial(jax.jit, backend=backend, static_argnums=(2,))
def smc(key, sigmas, n_particles):
    potential_fn = lambda z: jnp.log(jnp.sum(z ** 2) < 1)
    gen_observation_potential = PotentialKernel(potential_fn)
    init_observation_potential = InitPotentialKernel(potential_fn)

    transition_kernel = TransitionKernel(phi, sv * jnp.eye(D))
    initial_model = InitialModel(m0, chol0)

    if not use_sequential:
        qt = QtModel(sigmas)
        result = smoothing(key, qt, qt, transition_kernel, gen_observation_potential,
                           init_observation_potential, initial_model, N=n_particles, resampling_method=systematic)
        return result.ells[-1], (result.trajectories, result.origins)
    else:
        (ancestors, trajectories), ells = seq_smoothing(T, key, transition_kernel, gen_observation_potential,
                                                        initial_model, init_observation_potential, n_particles,
                                                        M=n_particles,
                                                        do_backward_pass=True)
        return ells[-1], (trajectories, ancestors)


@partial(jax.jit, backend=backend)
def learning_loop(key, init_sigmas):
    potential_fn = lambda z: jnp.log(jnp.sum(z ** 2) < 1)
    gen_observation_potential = PotentialKernel(potential_fn)
    init_observation_potential = InitPotentialKernel(potential_fn)

    transition_kernel = TransitionKernel(phi, sv * jnp.eye(D))
    initial_model = InitialModel(m0, chol0)
    opt_state_init = optimizer.init_fn(init_sigmas)

    def loss(sig, op_key):
        qt = QtModel(sig)
        return loss_fn(op_key, qt, qt, transition_kernel, gen_observation_potential,
                       init_observation_potential, initial_model, M)

    def body(carry, op_key):
        i, opt_state = carry
        value, grads = jax.value_and_grad(loss, 0)(optimizer.params_fn(opt_state), op_key)
        grads = jax.tree_map(lambda z: jnp.nan_to_num(z), grads)
        out = i + 1, optimizer.update_fn(i, grads, opt_state)
        return out, None

    all_keys = jax.random.split(key, n_steps)
    (_, final_state), _ = jax.lax.scan(body, (0, opt_state_init), all_keys)
    return optimizer.params_fn(final_state)


def run_experiment():
    sigmas = jnp.ones((T, D)) / D
    some_output, _ = smc(jax_key, sigmas, M)
    some_output.block_until_ready()

    if DO_LEARN and not use_sequential:
        learning_key, key = jax.random.split(jax_key, 2)
        sigmas = learning_loop(learning_key, sigmas)

    tic = time.time()
    if DO_LEARN and not use_sequential:
        learning_key, key = jax.random.split(jax_key, 2)
        again = learning_loop(learning_key, sigmas)
        again.block_until_ready()
    toc = time.time()

    print("Learning time", toc - tic)
    plt.plot(sigmas)
    plt.show()
    _ell, _ = smc(jax_key, sigmas, N)
    _ell.block_until_ready()

    @partial(jax.jit, backend=backend)
    def loop_body(i, op_key):
        def _print_fun(j, _):
            # Handmade progress bar function
            if j + 1 == B:
                print(f"\rIteration {j + 1}/{B}", end="\n", flush=False)
            else:
                print(f"\rIteration {j + 1}/{B}", end="", flush=True)

        ell, (trajs, _) = smc(op_key, sigmas, N)
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
