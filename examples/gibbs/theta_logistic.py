"""
Theta-logistic Gibbs sampling experiment design
"""
import os
import time
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.host_callback import id_tap
from matplotlib import pyplot as plt
from parsmooth.parallel import ieks
from parsmooth.utils import MVNormalParameters
from statsmodels.tsa.stattools import acf

from examples.models.theta_logistic import transition_function, observation_function, ObservationPotential, \
    TransitionKernel, InitialModel, InitObservationPotential
from parallel_ps.base import PyTree, UnivariatePotentialModel, GaussianDensity
from parallel_ps.core.resampling import multinomial
from parallel_ps.parallel_smoother import smoothing as particle_smoothing
from parallel_ps.sequential import conditional_smoother
from parallel_ps.utils import mvn_loglikelihood

sns.set_theme()

# IMPORTS

# CONFIG
warnings.simplefilter("ignore")  # ignore tensorflow probability dtypes warnings.
backend = "cpu"
# Particle smoother config

N = 50  # Number of particles
B = 10 ** 5  # Number of time steps in the chain
BURN_IN = B // 10  # Discarded number of steps for stats
KALMAN_N_ITER = 1  # Number of iterations to find new proposal q_t during the sampling process
KALMAN_N_ITER_INIT = 25  # Number of iterations to find initial proposal q_t to start the sampling process

use_kalman = False  # use an iterated Kalman smoother to sample the proposals, otherwise, just Gaussian around obs
use_sequential = True  # use the sequential algorithm instead of the parallel one.
# Data
data = np.genfromtxt('../data/nutria.txt', delimiter=',').reshape(-1, 1)
T = data.shape[0]

# Parameters priors, ideally using distrax, but these are not supported yet...
x_prec_prior = tfp.distributions.Gamma(2., 1.)
y_prec_prior = tfp.distributions.Gamma(2., 1.)
tau0_prior = tfp.distributions.TruncatedNormal(0., 1., 0., 3.)
tau1_prior = tfp.distributions.TruncatedNormal(0., 1., 0., 3.)
tau2_prior = tfp.distributions.TruncatedNormal(0., 1., 0., 3.)
tau2_scale = 0.2  # scale used for the RMH proposal in tau2

# State priors
m0 = jnp.zeros((1,))
chol_P0 = jnp.eye(1)

# Other config
jax_seed = np.random.randint(0, 123456)
print("JAX SEED: ", jax_seed)
jax_key = jax.random.PRNGKey(jax_seed)


# Utility functions to compute posteriors
def gamma_posterior(dist: tfp.distributions.Gamma, xs):
    return tfp.distributions.Gamma(dist.concentration + 0.5 * xs.shape[0],
                                   dist.rate + 0.5 * jnp.sum(xs ** 2))


def truncated_normal_posterior(dist: tfp.distributions.TruncatedNormal, xs, s=1.):
    n = xs.shape[0]
    pr0 = 1. / dist.scale ** 2  # prior precision
    prd = n / s ** 2  # data precision
    var = 1. / (pr0 + prd)  # posterior variance
    loc = var * (pr0 * dist.loc + prd * xs.mean())
    return tfp.distributions.TruncatedNormal(loc, scale=np.sqrt(var), low=dist.low, high=dist.high)


# DEFINE nu_t and q_t

class NutModel(UnivariatePotentialModel):
    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        return mvn_loglikelihood(particles, *parameter)


def spec_iks(tau0, tau1, tau2, Q, R, smoother_init, n_iter):
    x0 = MVNormalParameters(m0, chol_P0 @ chol_P0.T)
    partial_transition_function = partial(transition_function, tau_0=tau0, tau_1=tau1, tau_2=tau2)
    return ieks(x0, data, partial_transition_function, Q, observation_function, R,  # noqa
                smoother_init, n_iter, propagate_first=False)


@partial(jax.jit, static_argnums=(6,))
def cdsmc(key, x_prec, y_prec, tau0, tau1, tau2, n_iter, smoother_init=None, traj=None):
    """
    Run a conditional SMC (sequential!!!)

    Parameters
    ----------
    key: ...
    x_prec, y_prec, tau0, tau1, tau2: float
        ...
    n_iter:
        Number of iterations for the kalman smoother (if used at all)
    smoother_init
    traj: ...

    Returns
    -------
    ...
    """
    sample_key, select_key = jax.random.split(key)

    q_2 = 1 / x_prec
    r_2 = 1 / y_prec
    q = q_2 ** 0.5
    r = r_2 ** 0.5
    chol_Q = jnp.array([[q]])
    chol_R = jnp.array([[r]])
    # id_print(jnp.nan)

    gen_observation_potential = ObservationPotential(chol_R, data[1:])
    init_observation_potential = InitObservationPotential(chol_R, data[0])
    transition_kernel = TransitionKernel(tau0, tau1, tau2, chol_Q)
    initial_model = InitialModel(m0, chol_P0)

    # Even if use_sequential, we initialise gibbs with our method (easier to code, but no impact on experiments)

    if not use_sequential:
        if not use_kalman:
            std_dev = (q_2 + r_2) ** 0.5
            q_t = GaussianDensity(jnp.atleast_2d(data), std_dev * jnp.ones((T, 1, 1)))
            next_smoother_init = None

        else:
            if smoother_init is None:
                smoother_init = MVNormalParameters(data,
                                                   1e-1 * jnp.ones((T, 1, 1)))

            R = chol_R @ chol_R.T
            Q = chol_Q @ chol_Q.T

            next_smoother_init = spec_iks(tau0, tau1, tau2, Q, R, smoother_init, n_iter)

            kalman_mean, kalman_covs = next_smoother_init
            kalman_chols = kalman_covs ** 0.5  # Achtung: This only works because the state has one dimension !!!!
            q_t = GaussianDensity(kalman_mean, kalman_chols)
        nu_t = q_t
        samples = particle_smoothing(key, q_t, nu_t, transition_kernel, gen_observation_potential,
                                     initial_model, init_observation_potential, multinomial, N,
                                     conditional_trajectory=traj)

        ancestors = samples.origins
        trajectories = samples.trajectories
        idx = jax.random.randint(select_key, (), 0, N)
        origins = ancestors[:, idx]
        next_trajectory = jax.tree_map(lambda z: z[:, idx], trajectories)
    else:


        std_dev = (q_2 + r_2) ** 0.5
        q_t = GaussianDensity(jnp.atleast_2d(data), std_dev * jnp.ones((T, 1, 1)))
        next_smoother_init = None
        qt_sample_key, csmc_key = jax.random.split(sample_key)
        if traj is None:
            traj = q_t.sample(qt_sample_key, 1)[:, 0]
        origins, next_trajectory = conditional_smoother(T, traj, csmc_key, transition_kernel,
                                                        gen_observation_potential, initial_model,
                                                        init_observation_potential, N)
    # id_print(next_trajectory, what="next_traj")
    return next_trajectory, origins, next_smoother_init


@jax.jit
def sample_params(key, tau0, tau1, tau2, sampled_traj):
    """
    Sample from the posterior of the parameters of the theta-logistic model given a sampling trajectory and the data.
    """
    # Get keys for each parameter

    x_prec_key, y_prec_key, tau01_key, tau2_key = jax.random.split(key, 4)

    # compute the residuals
    xs = sampled_traj[:, 0]
    diff_xs = jnp.diff(xs)
    ys = data[:, 0]
    delta_x = xs[1:] - transition_function(xs[:-1], tau0, tau1, tau2)
    delta_y = ys - observation_function(xs)

    # compute the posterior distributions for the precision
    x_prec_post = gamma_posterior(x_prec_prior, delta_x)
    y_prec_post = gamma_posterior(y_prec_prior, delta_y)

    # sample precisions
    x_prec = x_prec_post.sample(seed=x_prec_key)
    y_prec = y_prec_post.sample(seed=y_prec_key)

    # sample using a RMH step
    tau2_key1, tau2_key2 = jax.random.split(tau2_key)
    u = jax.random.uniform(tau2_key1)
    tau2_prop = tau2 + tau2_scale * jax.random.normal(tau2_key2)
    new_deltaX = xs[1:] - transition_function(xs[:-1], tau0, tau1, tau2_prop)
    log_prob = 0.5 * x_prec * (jnp.sum(delta_x ** 2) - jnp.sum(new_deltaX ** 2))
    log_prob = log_prob + tau2_prior.log_prob(tau2_prop) - tau2_prior.log_prob(tau2)
    tau2 = jax.lax.cond(jnp.log(u) < log_prob, lambda _: tau2_prop, lambda _: tau2, None)
    # return x_prec, y_prec, 1., 1., tau2
    # Jointly sample from tau1 and tau2
    features = jnp.stack([jnp.ones((T - 1,)), -jnp.exp(tau2 * xs[:-1])], axis=-1)
    beta_ols, _, rank, singular = jnp.linalg.lstsq(features, diff_xs)  # least squares solution
    prior_mean = jnp.array([tau0_prior.loc, tau1_prior.loc])
    prior_prec = jnp.diag(jnp.array([1 / tau0_prior.scale ** 2, 1 / tau1_prior.scale ** 2]))

    xtx = features.T @ features
    post_prec = prior_prec + x_prec * xtx

    post_cov = jnp.linalg.inv(post_prec)
    post_mean = prior_prec @ prior_mean + x_prec * post_cov @ xtx @ beta_ols
    post_chol = jnp.linalg.cholesky(post_cov)

    def tau01_sample_loop(carry):
        _, _, op_key = carry
        next_key, sample_key = jax.random.split(op_key, 2)
        eps = jax.random.normal(sample_key, shape=(2,))
        prop = post_mean + post_chol @ eps
        return prop[0], prop[1], next_key

    def cond_fun(carry):
        return (carry[0] <= 0) | (carry[1] <= 0)

    tau0, tau1, _ = jax.lax.while_loop(cond_fun,
                                       tau01_sample_loop,
                                       (-1., -1., tau01_key)
                                       )

    return x_prec, y_prec, tau0, tau1, tau2


@partial(jax.jit, static_argnums=(1,), backend=backend)
def gibbs_routine(rng_key, n_iter):
    def count_rejuvenate(origins):
        return 0. + (origins != 0)

    init_key, rng_key = jax.random.split(rng_key)
    init_x_prec_key, init_y_prec_key, init_tau0_key, init_tau1_key, init_tau2_key = jax.random.split(init_key, 5)

    init_x_prec = x_prec_prior.sample(seed=init_x_prec_key)
    init_y_prec = y_prec_prior.sample(seed=init_y_prec_key)
    init_tau0 = tau0_prior.sample(seed=init_tau0_key)
    init_tau1 = tau1_prior.sample(seed=init_tau1_key)
    init_tau2 = tau2_prior.sample(seed=init_tau2_key)

    init_traj, _, smoother_init = cdsmc(init_key, init_x_prec, init_y_prec, init_tau0, init_tau1, init_tau2,
                                        KALMAN_N_ITER_INIT, None, None)

    def body(carry, inps):

        i, curr_key = inps

        def _print_fun(j, _):
            # Handmade progress bar function
            if j + 1 == n_iter:
                print(f"\rIteration {j + 1}/{n_iter}", flush=True)
            else:
                print(f"\rIteration {j + 1}/{n_iter}", end="", flush=True)

        id_tap(_print_fun, i)

        param_sampling_key, traj_sampling_key = jax.random.split(curr_key)
        curr_x_prec, curr_y_prec, curr_tau0, curr_tau1, curr_tau2, curr_traj, curr_smoother_init = carry
        next_x_prec, next_y_prec, next_tau0, next_tau1, next_tau2 = sample_params(
            param_sampling_key, curr_tau0, curr_tau1, curr_tau2, curr_traj)

        next_traj, origins, next_smoother_init = cdsmc(traj_sampling_key, next_x_prec, next_y_prec, next_tau0,
                                                       next_tau1, next_tau2,
                                                       KALMAN_N_ITER, curr_smoother_init, curr_traj)
        next_carry = next_x_prec, next_y_prec, next_tau0, next_tau1, next_tau2, next_traj, next_smoother_init
        rejuvenated = count_rejuvenate(origins)
        save = next_x_prec, next_y_prec, next_tau0, next_tau1, next_tau2, next_traj, rejuvenated
        return next_carry, save

    keys = jax.random.split(rng_key, n_iter)
    carry_init = init_x_prec, init_y_prec, init_tau0, init_tau1, init_tau2, init_traj, smoother_init
    _, (x_precs, y_precs, tau0s, tau1s, tau2s, trajs, rejuvenated_stats) = jax.lax.scan(body, carry_init,
                                                                                        (jnp.arange(n_iter), keys), )
    return x_precs, y_precs, tau0s, tau1s, tau2s, trajs, rejuvenated_stats


init_key, gibbs_key = jax.random.split(jax_key)

# # Compile the routine ahead of time to not count the impact of compilation time.
# temp, *_ = gibbs_routine(gibbs_key, B)
# temp.block_until_ready()

tic = time.time()
(x_prec_samples, y_prec_samples, tau0_samples, tau1_samples, tau2_samples, traj_samples,
 rejuvenated_logs) = gibbs_routine(gibbs_key, B)
x_prec_samples.block_until_ready()
toc = time.time()

x_prec_samples = x_prec_samples[BURN_IN:]
y_prec_samples = y_prec_samples[BURN_IN:]
tau0_samples = tau0_samples[BURN_IN:]
tau1_samples = tau1_samples[BURN_IN:]
tau2_samples = tau2_samples[BURN_IN:]
rejuvenated_logs = rejuvenated_logs[BURN_IN:]
traj_samples = traj_samples[BURN_IN:]

os.makedirs("./output", exist_ok=True)
np.savez(f"./output/theta-logistic-experiment-{use_kalman}-{use_sequential}", x_prec_samples=x_prec_samples,
         y_prec_samples=y_prec_samples,
         tau0_samples=tau0_samples, tau1_samples=tau1_samples, tau2_samples=tau2_samples,
         rejuvenated_logs=rejuvenated_logs.mean(0),
         traj_samples=traj_samples)

plt.plot(np.arange(T), rejuvenated_logs.mean(0))
plt.title(f"Update rate per time step, run time: {toc - tic:.0f}s, n iter: {B}")
plt.ylim(0, 1)
plt.show()

sns.jointplot(x=tau1_samples, y=tau0_samples)
plt.show()

sns.jointplot(x=tau2_samples, y=tau0_samples)
plt.show()

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 10))


def plot_acf(arr, ax, label):
    acf_data = acf(arr, nlags=100)
    ax.plot(acf_data, label=label)
    ax.set_xlabel("lag")
    ax.set_ylim(0, 1)
    ax.legend()


plot_acf(x_prec_samples, axes[1, 0], r"$1/\sigma_X^2$")
plot_acf(y_prec_samples, axes[1, 1], r"$1/\sigma_Y^2$")
plot_acf(tau0_samples, axes[0, 0], r"$\tau_0$")
plot_acf(tau1_samples, axes[0, 1], r"$\tau_1$")
plot_acf(tau2_samples, axes[0, 2], r"$\tau_2$")
plot_acf(traj_samples[:, 0, 0], axes[1, 2], r"$x_0$")

plt.show()
