""" Cox model """

import chex
import jax.random
import numpy as np
import scipy.stats as stats
from jax import numpy as jnp
from jax.scipy.stats import poisson

from parallel_ps.base import PyTree, ConditionalDensityModel, BivariatePotentialModel, UnivariatePotentialModel, \
    DensityModel
from parallel_ps.utils import mvn_loglikelihood


class TransitionKernel(ConditionalDensityModel):
    """
    X_t = N(mu + rho * (X_{t-1} - mu), sigma^2)
    """

    def __init__(self, mu, sigma, rho):
        super(TransitionKernel, self).__init__((mu, sigma, rho),
                                               (False, False, False))

    def sample(self, key: chex.PRNGKey, x_t_1: chex.ArrayTree, parameter: PyTree) -> chex.ArrayTree:
        mu, sigma, rho = parameter
        mean = mu + rho * (x_t_1 - mu)
        eps = jax.random.normal(key, x_t_1.shape)
        return mean + sigma * eps

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mu, sigma, rho = parameter
        mean = mu + rho * (x_t_1 - mu)
        return mvn_loglikelihood(x_t, mean, jnp.atleast_1d(sigma), is_diag=True)


class ObservationKernel(BivariatePotentialModel):
    """
    Y_t \sim V_t~N(0, diag(X_t / 2))
    """

    def __init__(self, ys):
        super(ObservationKernel, self).__init__(ys, True)

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, y_t: PyTree) -> jnp.ndarray:
        rate = jnp.exp(x_t[..., 0])
        return poisson.logpmf(y_t, rate)


class InitObservationPotential(UnivariatePotentialModel):
    def __init__(self, y_t):
        super(UnivariatePotentialModel, self).__init__(y_t, False)

    def log_potential(self, x_t: chex.ArrayTree, y_t: PyTree) -> jnp.ndarray:
        rate = jnp.exp(x_t[..., 0])
        return poisson.logpmf(y_t, rate)


class InitialModel(DensityModel):
    def __init__(self, mu, sigma, rho):
        super(DensityModel, self).__init__((mu, sigma, rho),
                                           (False, False, False))

    def log_potential(self, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mu, sigma, rho = parameter
        mu = jnp.atleast_1d(mu)
        chol = jnp.atleast_1d(sigma / (1. - rho ** 2) ** 0.5)
        return mvn_loglikelihood(x_t, mu, chol, is_diag=True)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        mu, sigma, rho = self.parameters
        chol = sigma / (1 - rho ** 2) ** 0.5

        eps = jax.random.normal(key, (N, 1))
        return mu + chol * eps


def get_data(mu, rho, sigma, T):
    x = mu + sigma * np.random.randn() / (1 - rho ** 2) ** 0.5
    y = stats.poisson.rvs(np.exp(x))

    xs = np.empty((T, 1))
    ys = np.empty((T,), dtype=int)

    xs[0, 0] = x
    ys[0] = y
    x_normals = np.random.randn(T - 1)

    for t in range(T - 1):
        x = mu + rho * (x - mu) + sigma * x_normals[t]
        y = stats.poisson.rvs(np.exp(x))
        xs[t + 1, 0] = x
        ys[t + 1] = y

    return xs, ys
