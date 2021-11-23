import os

import chex
import jax.random
import numpy as np
import quandl as quandl
from jax import numpy as jnp

from parallel_ps.base import PyTree, ConditionalDensityModel, BivariatePotentialModel, UnivariatePotentialModel, \
    DensityModel
from parallel_ps.utils import mvn_loglikelihood


class TransitionKernel(ConditionalDensityModel):
    """
    X_t = mu + diag(\phi)(X_{t-1} - mu) + U_t,   U_t~N(0, Q)
    """

    def __init__(self, mu, phi, chol):
        parameters = mu, phi, chol
        super(TransitionKernel, self).__init__(parameters, (False,) * 3)

    def sample(self, key: chex.PRNGKey, x_t_1: chex.ArrayTree, parameter: PyTree) -> chex.ArrayTree:
        mu, phi, chol = parameter
        mean = mu[None:] + phi[None, :] * (x_t_1 - mu[None, :])
        eps = jax.random.normal(key, x_t_1.shape)
        return mean + jnp.einsum("ij,kj->ki", chol, eps)

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mu, phi, chol = parameter
        mean = mu + phi * (x_t_1 - mu)
        return mvn_loglikelihood(x_t, mean, chol)


class ObservationKernel(BivariatePotentialModel):
    """
    Y_t \sim V_t~N(0, diag(X_t / 2))
    """

    def __init__(self, ys):
        super(ObservationKernel, self).__init__(ys, True)

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, yt: PyTree) -> jnp.ndarray:
        scale = jnp.exp(0.5 * x_t_1)
        return mvn_loglikelihood(yt, x_t, scale, is_diag=True)


class InitObservationPotential(UnivariatePotentialModel):
    def __init__(self, y_t):
        super(UnivariatePotentialModel, self).__init__(y_t, False)

    def log_potential(self, x_t: chex.ArrayTree, y_t: PyTree) -> jnp.ndarray:
        scale = jnp.exp(0.5 * x_t)
        return mvn_loglikelihood(y_t, x_t, scale, is_diag=True)


class InitialModel(DensityModel):
    def __init__(self, m, chol):
        super(DensityModel, self).__init__((m, chol),
                                           (False, False))

    def log_potential(self, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        m0, chol_P0 = parameter
        return mvn_loglikelihood(x_t, m0, chol_P0)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        m0, chol_P0 = self.parameters
        eps = jax.random.normal(key, (N, m0.shape[0]))
        return m0[None, :] + jnp.einsum("ij,kj->ki", chol_P0, eps)


def _solve_discrete_lyapunov_diagonal(phi, chol):
    """
    Solves CC^T = phi C C^T phi^T + chol chol^T when phi is diagonal.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.linalg as linalg
    >>> d = 3
    >>> phi = np.random.rand(d)
    >>> chol = np.random.rand(d, d)
    >>> chol[np.triu_indices(d, 1)] = 0.
    >>> actual_chol = _solve_discrete_lyapunov_diagonal(phi, chol)
    >>> actual = actual_chol @ actual_chol.T
    >>> expected = linalg.solve_discrete_lyapunov(np.diag(phi), chol @ chol.T)
    >>> np.allclose(actual, expected)
    True
    """

    cov = chol @ chol.T
    matrix = jnp.outer(phi, phi)

    temp = cov / (1. - matrix)
    return jnp.linalg.cholesky(temp)


def get_stationary_distribution(m, phi, chol):
    stationary_chol = _solve_discrete_lyapunov_diagonal(phi, chol)
    return m, stationary_chol


def get_data(currencies, start_date, end_date, api_key=None):
    """
    In the paper we used `get_data(["USD", "CAD", "CHF"], "2019-11-01", "2021-11-01")`

    """
    path = f"../data/{currencies}{start_date}{end_date}.npy"
    if os.path.exists(path):
        return np.load(path)

    if api_key is not None:
        quandl.ApiConfig.api_key = api_key

    d = quandl.get([f"ECB/EUR{cur}" for cur in currencies], start_date=start_date, end_date=end_date)
    y = np.log(d).diff(axis=0).dropna()
    y = y.values
    np.save(path, y)
    return y


