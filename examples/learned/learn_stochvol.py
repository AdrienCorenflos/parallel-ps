"""
Bearings-only smoothing experiment design
"""
# IMPORTS
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.host_callback import id_tap
from jax.experimental.optimizers import adam
from matplotlib import pyplot as plt

from examples.models.stochvol import InitObservationPotential, InitialModel, TransitionKernel, get_data, \
    ObservationKernel, get_stationary_distribution
from parallel_ps.base import PyTree, DensityModel
from parallel_ps.parallel_smoother import smoothing, loss_fn
from parallel_ps.sequential import smoothing as seq_smoothing
from tests.lgssm import mvn_loglikelihood

# CONFIG
DO_RUN = True
data = get_data(["USD", "CAD", "CHF"], "2019-11-01", "2021-11-01")

# PS config
backend = "gpu"
N = 50  # Number of particles
B = 4  # number of parallel PFs

# Learning config
N_STEPS = 10_000  # Number of time steps in the chain

use_sequential = False  # use the sequential algorithm instead of the parallel one.
T, D = data.shape

# Initial parameters
mu_prior = tfp.distributions.Normal(0., 1.)
diag_chol_prior = tfp.distributions.InverseGamma(3., 0.5)
chol_lower_prior = tfp.distributions.Normal(0., 0.1)
phi_prior = tfp.distributions.Uniform(-0.5, 0.5)

# Other config
jax_seed = 42
jax_key = jax.random.PRNGKey(jax_seed)
learning_rate = 1e-4
optimizer = adam(learning_rate)


class QtModel(DensityModel):
    def __init__(self, mu, chol):
        super(QtModel, self).__init__((mu, chol), (False, False), T=T)

    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mu, chol = parameter
        return mvn_loglikelihood(particles, mu, chol)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        mu, chol = self.parameters
        normals = jax.random.normal(key, (self.T, N, mu.shape[0]))
        return mu[None, None, :] + jnp.einsum("ij,...kj->...ki", chol, normals)


@partial(jax.jit, backend=backend)
def smc(key, opt_state, n_iter):
    gen_observation_potential = ObservationKernel(data[1:])
    init_observation_potential = InitObservationPotential(data[0])

    def loss(params):
        keys = jax.random.split(key, B)
        losses = jax.vmap(_loss, [None, 0])(params, keys)
        return jnp.mean(losses)

    def _loss(params, loss_key):
        mu, chol, phi = params
        chol = jnp.zeros((D, D)).at[jnp.tril_indices(D)].set(chol)
        transition_kernel = TransitionKernel(mu, phi, chol)
        stationary_mu, stationary_chol = get_stationary_distribution(mu, phi, chol)
        initial_model = InitialModel(stationary_mu, stationary_chol)

        if not use_sequential:
            qt = QtModel(stationary_mu, stationary_chol)
            result = loss_fn(loss_key, qt, qt, transition_kernel, gen_observation_potential,
                             init_observation_potential, initial_model, N=N)
            return result
        else:
            (_, _), ells = seq_smoothing(T, loss_key, transition_kernel, gen_observation_potential, initial_model,
                                         init_observation_potential, N, do_backward_pass=False)
            return -ells[-1]

    value, grads = jax.value_and_grad(loss)(optimizer.params_fn(opt_state))
    grads = jax.tree_map(lambda z: jnp.nan_to_num(z), grads)
    return -value, optimizer.update_fn(n_iter, grads, opt_state)


@partial(jax.jit, static_argnums=(1,), backend=backend)
def learning_routine(rng_key, n_iter):
    rng_key, mu_key, chol_key, chol_diag_key, phi_key = jax.random.split(rng_key, 5)

    init_mu = mu_prior.sample(sample_shape=(D,), seed=mu_key)
    init_phi = phi_prior.sample(sample_shape=(D,), seed=phi_key)
    init_chol = chol_lower_prior.sample(sample_shape=(D, D), seed=chol_key)
    init_chol = init_chol.at[jnp.diag_indices(D)].set(diag_chol_prior.sample(sample_shape=(D,), seed=chol_diag_key))
    init_chol = init_chol[jnp.tril_indices(D)]

    opt_state_init = optimizer.init_fn((init_mu, init_chol, init_phi))
    sample_key, rng_key = jax.random.split(rng_key)

    ell_init, _ = smc(sample_key, opt_state_init, 0)

    def body(carry, inps):
        opt_state, previous_ell = carry
        i, curr_key = inps

        def _print_fun(arg, _):
            # Handmade progress bar function
            j, ell_val, previous_ell_val = arg
            txt = f"\rIteration {j + 1}/{n_iter}, current ell: {ell_val:2f}, previous ell: {previous_ell_val:2f}"
            if j + 1 == n_iter:
                print(txt, flush=True)
            else:
                print(txt, end="", flush=True)

        ell, opt_state = smc(curr_key, opt_state, i)
        id_tap(_print_fun, (i, ell, previous_ell))

        carry = opt_state, ell
        return carry, ell

    keys = jax.random.split(rng_key, n_iter)
    carry_init = opt_state_init, ell_init
    (final_opt_state, *_), ells = jax.lax.scan(body, carry_init, (jnp.arange(n_iter), keys), )
    return optimizer.params_fn(final_opt_state), ells


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def run_experiment():
    (mu_final, chol_final, phi_final), ells = learning_routine(jax_key, N_STEPS)
    print()
    print(mu_final)
    print()
    print(jnp.zeros((D, D)).at[jnp.tril_indices(D)].set(chol_final))
    print()
    print(phi_final)

    plt.plot(ells)
    plt.show()


if DO_RUN:
    run_experiment()
