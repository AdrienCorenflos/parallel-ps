"""
Bearings-only smoothing experiment design
"""
# IMPORTS
import time
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.host_callback import id_tap
from jax.experimental.optimizers import sgd
from matplotlib import pyplot as plt

from examples.models.stochvol import InitObservationPotential, InitialModel, TransitionKernel, get_data, \
    ObservationKernel, get_stationary_distribution
from parallel_ps.base import PyTree, UnivariatePotentialModel, DensityModel
from parallel_ps.parallel_smoother import smoothing
from parallel_ps.sequential import smoothing as seq_smoothing
from tests.lgssm import mvn_loglikelihood

# CONFIG
DO_RUN = True
data = get_data(["USD", "CAD", "CHF"], "2019-11-01", "2021-11-01")

# PS config
backend = "cpu"
N = 25  # Number of particles

# PMMH config
N_CHAINS = 10 ** 5  # Number of time steps in the chain
BURN_IN = N_CHAINS // 10  # Discarded number of steps for stats
N_GRADIENT_STEPS = N_CHAINS // 10  # Warmup of the proposals

use_sequential = True  # use the sequential algorithm instead of the parallel one.

T, D = data.shape

# Initial parameters
mu_prior = tfp.distributions.Normal(0., 1.)
diag_chol_prior = tfp.distributions.InverseGamma(2., 1.)
chol_lower_prior = tfp.distributions.Normal(0., 1.)
phi_prior = tfp.distributions.Uniform(-1., 1.)

# MCMC parameters
mcmc_scale = 0.2

# Other config
jax_seed = 0
jax_key = jax.random.PRNGKey(jax_seed)
learning_rate = 1e-3
optimizer = sgd(learning_rate)


def _make_init_weights_and_bias(mu, chol, phi, _):
    # conditional linearization
    stationary_mean, stationary_chol = get_stationary_distribution(mu, phi, chol)

    mu_weights = jnp.zeros((D, D))
    mu_bias = jnp.zeros((D,))

    phi_weights = jnp.zeros((D, D))
    phi_bias = jnp.zeros((D,))

    chol_weights = jnp.zeros((D, (D * (D + 1)) // 2))
    chol_bias = jnp.zeros((D,))

    chol_weights_out = jnp.zeros(((D * (D + 1)) // 2, 3 * D))
    diag_part = jnp.diag_indices(D)
    stationary_diag = stationary_chol[jnp.diag_indices(D)]
    stationary_chol = stationary_chol.at[diag_part].set(jnp.log(stationary_diag))
    chol_bias_out = stationary_chol[jnp.tril_indices(D)]

    mean_bias = stationary_mean
    mean_weight = jnp.zeros((D, 3 * D))
    mu_params = mu_weights, mu_bias
    chol_params = chol_weights, chol_bias
    phi_params = phi_weights, phi_bias

    return mu_params, chol_params, phi_params, mean_weight, mean_bias, chol_weights_out, chol_bias_out


# DEFINE nu_t and q_t
def _make_mvn_proposal(params, mu, chol, phi):
    tril = jnp.tril_indices(D)
    diag = jnp.diag_indices(D)

    mu_params, chol_params, phi_params, weights_mean, weights_bias, chol_weights_out, chol_bias_out = params
    mu_weights, mu_bias = mu_params
    chol_weights, chol_bias = chol_params
    phi_weights, phi_bias = phi_params

    mu_part = jax.nn.sigmoid(mu_weights @ mu + mu_bias)
    chol_part = jax.nn.sigmoid(chol_weights @ chol[tril] + chol_bias)
    phi_part = jax.nn.sigmoid(phi_weights @ phi + phi_bias)

    mixing_out = jnp.concatenate([mu_part, chol_part, phi_part], 0)

    mean = weights_mean @ mixing_out + weights_bias
    chol_tril = chol_weights_out @ mixing_out + chol_bias_out
    chol_out = jnp.zeros_like(chol)
    chol_out = chol_out.at[tril].set(chol_tril)
    chol_out = chol_out.at[diag].set(jnp.exp(jnp.diag(chol_out)))
    return mean, chol_out


class QtModel(DensityModel):
    def __init__(self, parameters, mu, chol, phi):
        batched = jax.tree_map(lambda z: True, parameters)
        super(QtModel, self).__init__((parameters, mu, chol, phi), (batched, False, False, False), T=T)

    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        params, mu, chol, phi = parameter
        proposal_mean, proposal_chol = _make_mvn_proposal(params, mu, chol, phi)
        return mvn_loglikelihood(particles, proposal_mean, proposal_chol)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        params, mu, chol, phi = self.parameters
        proposal_means, proposal_chols = jax.vmap(_make_mvn_proposal, in_axes=[0, None, None, None])(params, mu,
                                                                                                     chol, phi)
        normals = jax.random.normal(key, (self.T, N, mu.shape[0]))
        return proposal_means[:, None, :] + jnp.einsum("...ij,...kj->...ki", proposal_chols, normals)


class NutModel(UnivariatePotentialModel):
    def __init__(self, parameters, mu, chol, phi):
        batched = jax.tree_map(lambda z: True, parameters)
        super(NutModel, self).__init__((parameters, mu, chol, phi),
                                       (batched, False, False, False))

    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        params, mu, chol, phi = parameter
        proposal_mean, proposal_chol = _make_mvn_proposal(params, mu, chol, phi)
        return mvn_loglikelihood(particles, proposal_mean, proposal_chol)


# define the routine

@partial(jax.jit, backend=backend)
def smc(key, mu, chol, phi, opt_state, gradient_step):
    """
    Run a SMC

    Parameters
    ----------
    key: ...
    mu, chol, phi: float
        ...
    opt_state:
        ...
    gradient_step:
        ...
    Returns
    -------
    ...
    """
    stationary_mu, stationary_chol = get_stationary_distribution(mu, phi, chol)

    gen_observation_potential = ObservationKernel(data[1:])
    init_observation_potential = InitObservationPotential(data[0])
    transition_kernel = TransitionKernel(mu, phi, chol)

    initial_model = InitialModel(stationary_mu, stationary_chol)

    def loss(qt_params, nut_params):
        if not use_sequential:
            qt = QtModel(qt_params, mu, chol, phi)
            nut = NutModel(nut_params, mu, chol, phi)
            result = smoothing(key, qt, nut, transition_kernel, gen_observation_potential, init_observation_potential,
                               initial_model, N=N)
            return result.ells[-1]
        else:
            (_, _), ells = seq_smoothing(T, key, transition_kernel, gen_observation_potential, initial_model,
                                         init_observation_potential, N, do_backward_pass=False)
            return ells[-1]

    def if_grad_pass(_):
        value, grads = jax.value_and_grad(loss, [0, 1])(*optimizer.params_fn(opt_state))
        return value, optimizer.update_fn(0, grads, opt_state)

    def otherwise(_):
        value = loss(*optimizer.params_fn(opt_state))
        return value, opt_state

    cond = jnp.logical_and(gradient_step, jnp.logical_not(use_sequential))
    return jax.lax.cond(cond, if_grad_pass, otherwise, None)


@jax.jit
def rmh_step(key, mu, chol, phi, step, opt_state, prev_ell, prev_params_log_lik):
    # Get keys for each parameter
    triu = jnp.triu_indices(D, 1)
    tril = jnp.tril_indices(D, -1)
    diag_term = jnp.diag_indices(D)

    key, mu_key, chol_key, phi_key = jax.random.split(key, 4)
    mu_prop = mu + mcmc_scale * jax.random.normal(mu_key, mu.shape)
    chol_prop = chol + mcmc_scale * jax.random.normal(chol_key, chol.shape)
    chol_prop = chol_prop.at[triu].set(0.)
    phi_prop = phi + mcmc_scale * jax.random.normal(phi_key, mu.shape)

    gradient_step = step < N_GRADIENT_STEPS

    key, sample_key = jax.random.split(key, 2)
    ell, new_opt_state = smc(sample_key, mu_prop, chol_prop, phi_prop, opt_state, gradient_step)

    mu_log_lik = jnp.sum(jax.vmap(mu_prior.log_prob)(mu_prop))
    chol_log_lik = jnp.sum(jax.vmap(chol_lower_prior.log_prob)(chol_prop[tril]))
    chol_log_lik = chol_log_lik + jnp.sum(jax.vmap(diag_chol_prior.log_prob)(chol_prop[diag_term]))
    phi_log_lik = jnp.sum(jax.vmap(phi_prior.log_prob)(phi_prop))

    prop_params_log_lik = mu_log_lik + chol_log_lik + phi_log_lik

    acceptance_log_ratio = ell - prev_ell + prop_params_log_lik - prev_params_log_lik

    u = jax.random.uniform(key)
    p_accept = jnp.minimum(jnp.exp(acceptance_log_ratio), 1.)
    accept = u < p_accept

    def if_accept(_):
        return 1, p_accept, mu_prop, chol_prop, phi_prop, new_opt_state, ell, prop_params_log_lik

    def otherwise(_):
        return 0, p_accept, mu, chol, phi, new_opt_state, prev_ell, prev_params_log_lik

    return jax.lax.cond(accept, if_accept, otherwise, None)


@partial(jax.jit, static_argnums=(1,), backend=backend)
def sampling_routine(rng_key, n_iter):
    rng_key, mu_key, chol_key, chol_diag_key, phi_key = jax.random.split(rng_key, 5)

    init_mu = mu_prior.sample(sample_shape=(D,), seed=mu_key)
    init_phi = phi_prior.sample(sample_shape=(D,), seed=phi_key)
    init_chol = chol_lower_prior.sample(sample_shape=(D, D), seed=chol_key)
    init_chol = init_chol.at[jnp.triu_indices(D, 1)].set(0.)
    init_chol = init_chol.at[jnp.diag_indices(D)].set(diag_chol_prior.sample(sample_shape=(D,), seed=chol_diag_key))

    mu_log_lik = jnp.sum(jax.vmap(mu_prior.log_prob)(init_mu))
    chol_log_lik = jnp.sum(jax.vmap(chol_lower_prior.log_prob)(init_chol[jnp.tril_indices(D, -1)]))
    chol_log_lik = chol_log_lik + jnp.sum(jax.vmap(diag_chol_prior.log_prob)(init_chol[jnp.diag_indices(D)]))
    phi_log_lik = jnp.sum(jax.vmap(phi_prior.log_prob)(init_phi))

    init_params_log_lik = mu_log_lik + chol_log_lik + phi_log_lik

    init_qt_params = jax.vmap(_make_init_weights_and_bias, [None, None, None, 0])(init_mu, init_phi, init_chol, data)
    init_nut_params = jax.vmap(_make_init_weights_and_bias, [None, None, None, 0])(init_mu, init_phi, init_chol, data)
    opt_state_init = optimizer.init_fn((init_qt_params, init_nut_params))

    sample_key, rng_key = jax.random.split(rng_key)

    ell_init, _ = smc(sample_key, init_mu, init_chol, init_phi, opt_state_init, False)

    def body(carry, inps):
        mu, chol, phi, opt_state, ell, params_log_lik = carry
        i, curr_key = inps

        def _print_fun(j, _):
            # Handmade progress bar function
            if j + 1 == n_iter:
                print(f"\rIteration {j + 1}/{n_iter}", flush=True)
            else:
                print(f"\rIteration {j + 1}/{n_iter}", end="", flush=True)

        id_tap(_print_fun, i)
        accepted, p_accept, mu, chol, phi, opt_state, ell, params_log_lik = rmh_step(curr_key, mu,
                                                                                     chol, phi, i,
                                                                                     opt_state, ell,
                                                                                     params_log_lik)

        carry = mu, chol, phi, opt_state, ell, params_log_lik
        save = accepted, p_accept, mu, chol, phi
        return carry, save

    keys = jax.random.split(rng_key, n_iter)
    carry_init = init_mu, init_chol, init_phi, opt_state_init, ell_init, init_params_log_lik
    _, (all_accepted, all_p_accept, mus, chols, phis) = jax.lax.scan(body, carry_init,
                                                                     (jnp.arange(n_iter), keys), )
    return all_accepted, all_p_accept, mus, chols, phis


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def run_experiment():
    tic = time.time()
    all_accepted, all_p_accept, mus, chols, phis = sampling_routine(jax_key, N_CHAINS)
    toc = time.time()

    cummean_accepted = np.cumsum(all_accepted) / np.arange(1, N_CHAINS + 1)
    cummean_p_accept = np.cumsum(all_p_accept) / np.arange(1, N_CHAINS + 1)

    mus = mus[BURN_IN:]
    chols = chols[BURN_IN:]
    phis = phis[BURN_IN:]

    fig, ax = plt.subplots()
    ax.plot(all_accepted)
    ax.plot(all_p_accept)
    plt.show()


if DO_RUN:
    run_experiment()
