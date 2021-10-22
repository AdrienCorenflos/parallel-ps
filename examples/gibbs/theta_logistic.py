"""
Theta-logistic Gibbs sampling experiment design
"""
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

from examples.models.theta_logistic import transition_function, observation_function, ObservationPotential, \
    TransitionKernel, InitialModel, InitObservationPotential
from parallel_ps.base import DensityModel, PyTree, UnivariatePotentialModel, KalmanSmootherDensity
from parallel_ps.core.resampling import multinomial
from parallel_ps.parallel_smoother import smoothing as particle_smoothing
from parsmooth import MVNSqrt, FunctionalModel
from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing
from tests.lgssm import mvn_loglikelihood

sns.set_theme()

# IMPORTS

# CONFIG
warnings.simplefilter("ignore")  # ignore tensorflow probability dtypes warnings.
backend = "gpu"
# Particle smoother config

N = 50  # Number of particles
B = 10 ** 5  # Number of time steps in the chain
BURN_IN = B // 10  # Discarded number of steps for stats
KALMAN_N_ITER = 3

uniform = False  # use a uniform (random) grid to sample the proposals
min_uniform = 0.
max_uniform = 10.
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


# State priors
m0 = jnp.zeros((1,))
chol_P0 = jnp.eye(1)

# Other config
jax_seed = 123456
jax_key = jax.random.PRNGKey(jax_seed)


# DEFINE nu_t and q_t

class NutModel(UnivariatePotentialModel):
    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        return mvn_loglikelihood(particles, *parameter)


class UniformDensityModel(DensityModel):
    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        min_, max_ = parameter
        return 1 / (max_ - min_)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        min_, max_ = self.parameters
        uniforms = jax.random.uniform(key, (self.T, N, 1))
        return min_ + (max_ - min_) * uniforms


@jax.jit
def cdsmc(key, x_prec, y_prec, tau0, tau1, tau2, smoother_init=None, traj=None):
    """
    Run a conditional dSMC

    Parameters
    ----------
    key: ...
    x_prec, y_prec, tau0, tau1, tau2: float
        ...
    smoother_init
    traj: ...

    Returns
    -------
    ...
    """

    zero = jnp.zeros((1,))
    q = 1 / x_prec ** 0.5
    chol_Q = jnp.array([[q]])
    chol_R = jnp.array([[1 / y_prec ** 0.5]])

    if uniform:
        q_t = UniformDensityModel((min_uniform, max_uniform), (False, False))
        next_smoother_init = None

    else:
        if smoother_init is None:
            smoother_init = MVNSqrt(jnp.concatenate([jnp.zeros((1, 1)), data]),
                                    1e-1 * jnp.ones((T + 1, 1, 1)))
        kalman_transition_model = FunctionalModel(lambda x, q_val: transition_function(x, tau0, tau1, tau2) + q_val,
                                                  MVNSqrt(zero, chol_Q))
        kalman_observation_model = FunctionalModel(lambda x, r_val: observation_function(x) + r_val,
                                                   MVNSqrt(zero, chol_R))
        kalman_smoothing_solution = iterated_smoothing(data, MVNSqrt(m0, chol_P0), kalman_transition_model,
                                                       kalman_observation_model, extended, smoother_init,
                                                       criterion=lambda i, *_: i < KALMAN_N_ITER)
        next_smoother_init = kalman_smoothing_solution

        # Parsmooth considers that the first step is always a prediction, so we need to get rid of it via a vile hack.
        # It's fine as it is only the proposal q_t, but it's suboptimal. Need to fix this.
        q_t = KalmanSmootherDensity(*jax.tree_map(lambda z: z[1:], kalman_smoothing_solution))
    nu_t = q_t

    gen_observation_potential = ObservationPotential(chol_R, data[1:])
    init_observation_potential = InitObservationPotential(chol_R, data[0])
    samples = particle_smoothing(key, q_t, nu_t, TransitionKernel(tau0, tau1, tau2, chol_Q), gen_observation_potential,
                                 InitialModel(m0, chol_P0), init_observation_potential, multinomial, N,
                                 conditional_trajectory=traj)

    return samples, next_smoother_init


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
        return jnp.sum(origins != 0, axis=1)

    init_key, rng_key = jax.random.split(rng_key)
    init_x_prec_key, init_y_prec_key, init_tau0_key, init_tau1_key, init_tau2_key = jax.random.split(init_key, 5)

    init_x_prec = x_prec_prior.sample(seed=init_x_prec_key)
    init_y_prec = y_prec_prior.sample(seed=init_y_prec_key)
    init_tau0 = tau0_prior.sample(seed=init_tau0_key)
    init_tau1 = tau1_prior.sample(seed=init_tau1_key)
    init_tau2 = tau2_prior.sample(seed=init_tau2_key)

    init_trajs, smoother_init = cdsmc(init_key, init_x_prec, init_y_prec, init_tau0, init_tau1, init_tau2, None, None)

    traj_key, rng_key = jax.random.split(rng_key)
    init_traj = init_trajs.trajectories[:, jax.random.randint(traj_key, (), 0, N)]  # unconditional

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

        sample_key, select_key = jax.random.split(traj_sampling_key)

        traj_samples, next_smoother_init = cdsmc(sample_key, next_x_prec, next_y_prec, next_tau0, next_tau1, next_tau2,
                                                 curr_smoother_init, curr_traj)
        rejuvenated = count_rejuvenate(traj_samples.origins)
        next_traj = traj_samples.trajectories[:, jax.random.randint(select_key, (), 0, N - 1)]
        next_carry = next_x_prec, next_y_prec, next_tau0, next_tau1, next_tau2, next_traj, next_smoother_init
        save = next_x_prec, next_y_prec, next_tau0, next_tau1, next_tau2, next_traj, rejuvenated
        return next_carry, save

    keys = jax.random.split(gibbs_key, n_iter)
    carry_init = init_x_prec, init_y_prec, init_tau0, init_tau1, init_tau2, init_traj, smoother_init
    _, (x_precs, y_precs, tau0s, tau1s, tau2s, trajs, rejuvenated_stats) = jax.lax.scan(body, carry_init,
                                                                                        (jnp.arange(n_iter), keys), )
    return x_precs, y_precs, tau0s, tau1s, tau2s, trajs, rejuvenated_stats


init_key, gibbs_key = jax.random.split(jax_key)

tic = time.time()
x_prec_samples, y_prec_samples, tau0_samples, tau1_samples, tau2_samples, traj_samples, rejuvenated_logs = gibbs_routine(
    gibbs_key, B)
x_prec_samples.block_until_ready()
toc = time.time()

x_prec_samples = x_prec_samples[BURN_IN:]
y_prec_samples = y_prec_samples[BURN_IN:]
tau0_samples = tau0_samples[BURN_IN:]
tau1_samples = tau1_samples[BURN_IN:]
tau2_samples = tau2_samples[BURN_IN:]
rejuvenated_logs = rejuvenated_logs[BURN_IN:]
traj_samples = traj_samples[BURN_IN:]

np.savez("./output/theta-logistic-experiment", x_prec_samples=x_prec_samples, y_prec_samples=y_prec_samples,
         tau0_samples=tau0_samples, tau1_samples=tau1_samples, tau2_samples=tau2_samples,
         rejuvenated_logs=rejuvenated_logs,
         traj_samples=traj_samples)

plt.plot(np.arange(T), rejuvenated_logs.mean(0) / (N - 1))
plt.title(f"Update rate per time step, run time: {toc - tic:.2f}, n iter: {B}")
plt.show()

sns.jointplot(x=tau1_samples, y=tau0_samples)
plt.show()

sns.jointplot(x=tau2_samples, y=tau0_samples)
plt.show()
