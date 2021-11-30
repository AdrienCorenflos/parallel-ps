""" Hard Obstacle model """

import chex
import jax.random
from jax import numpy as jnp

from parallel_ps.base import PyTree, ConditionalDensityModel, BivariatePotentialModel, DensityModel, \
    UnivariatePotentialModel
from parallel_ps.utils import mvn_loglikelihood


class TransitionKernel(ConditionalDensityModel):
    """
    X_t = N(\phi(X_{t-1}), chol chol^T)
    """

    def __init__(self, phi, chol):
        super(TransitionKernel, self).__init__(chol, False)
        self._phi = phi

    def sample(self, key: chex.PRNGKey, x_t_1: chex.ArrayTree, chol: PyTree) -> chex.ArrayTree:
        mean = jax.vmap(self._phi)(x_t_1)
        eps = jax.random.normal(key, x_t_1.shape)
        return mean + jnp.einsum("ij,nj->ni", chol, eps)

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, chol: PyTree) -> jnp.ndarray:
        mean = self._phi(x_t_1)
        return mvn_loglikelihood(x_t, mean, chol, is_diag=False)


class PotentialKernel(BivariatePotentialModel):
    def __init__(self, ind_fn):
        self._ind_fn = ind_fn
        super(PotentialKernel, self).__init__(jnp.nan, False)

    def log_potential(self, _x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, _: PyTree) -> jnp.ndarray:
        return self._ind_fn(x_t)


class InitPotentialKernel(UnivariatePotentialModel):
    def __init__(self, ind_fn):
        self._ind_fn = ind_fn
        super(InitPotentialKernel, self).__init__(jnp.nan, False)

    def log_potential(self, x_t: chex.ArrayTree, _: PyTree) -> jnp.ndarray:
        return self._ind_fn(x_t)


class InitialModel(DensityModel):
    def __init__(self, mu, chol):
        super(DensityModel, self).__init__((mu, chol),
                                           (False, False))

    def log_potential(self, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mu, chol = parameter
        return mvn_loglikelihood(x_t, mu, chol, is_diag=False)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        mu, chol = self.parameters
        eps = jax.random.normal(key, (N, mu.shape[0]))
        return mu[None, :] + jnp.einsum("ij,nj->ni", chol, eps)
