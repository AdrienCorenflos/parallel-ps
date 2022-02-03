""" LGSSM model def """
import chex
import jax.numpy as jnp
import jax.random
import numpy as np

from parallel_ps.base import BivariatePotentialModel, PyTree
from parallel_ps.utils import mvn_loglikelihood


class LinearGaussianTransitionModel(BivariatePotentialModel):
    parameters: PyTree
    batched: PyTree

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        F, b, cholQ = parameter
        return mvn_loglikelihood(x_t, F @ x_t_1 + b, cholQ)


class LinearGaussianObservationModel(BivariatePotentialModel):
    parameters: PyTree
    batched: PyTree

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        H, c, cholR, y_t = parameter
        return mvn_loglikelihood(y_t, H @ x_t + c, cholR)


def get_data(m0, chol_P0, F, H, chol_R, chol_Q, b, c, T):
    """
    Parameters
    ----------
    m0: array_like
        initial_mean
    chol_P0: array_like
        initial cholesky cov
    F: array_like
        transition matrix
    H: array_like
        transition matrix
    chol_R: array_like
        observation model cholesky cov
    chol_Q: array_like
        noise cholesky cov
    b: array_like
        transition offset
    c: array_like
        observation offset
    T: int
        number of time steps

    Returns
    -------
    true_states: array_like
        array of true states
    observations: array_like
        array of observations
    """

    dim_y = H.shape[0]
    dim_x = F.shape[0]

    normals_x = np.random.randn(T, dim_x)
    normals_y = np.random.randn(T, dim_y)
    x = m0 + chol_P0 @ np.random.randn(dim_x)

    observations = np.empty((T, dim_y))
    true_states = np.empty((T + 1, dim_x))
    true_states[0] = x

    for i, (norm_x, norm_y) in enumerate(zip(normals_x, normals_y)):
        x = F @ x + chol_Q @ norm_x + b
        true_states[i + 1] = x
        y = H @ x + chol_R @ norm_y + c
        observations[i] = y

    return true_states, observations


def get_data_jax(k, m0, chol_P0, F, H, chol_R, chol_Q, b, c, T):
    """
    Parameters
    ----------
    k: chex.PRNGKey
        Sampling key
    m0: array_like
        initial_mean
    chol_P0: array_like
        initial cholesky cov
    F: array_like
        transition matrix
    H: array_like
        transition matrix
    chol_R: array_like
        observation model cholesky cov
    chol_Q: array_like
        noise cholesky cov
    b: array_like
        transition offset
    c: array_like
        observation offset
    T: int
        number of time steps

    Returns
    -------
    true_states: array_like
        array of true states
    observations: array_like
        array of observations
    """

    dim_y = H.shape[0]
    dim_x = F.shape[0]

    x_key, y_key, init_key = jax.random.split(k, 3)
    normals_x = jax.random.normal(x_key, (T, dim_x))
    normals_y = jax.random.normal(y_key, (T, dim_y))
    x0 = m0 + chol_P0 @ jax.random.normal(init_key, (dim_x,))

    def body(carry, inputs):
        x = carry
        n_x, n_y = inputs
        x = F @ x + chol_Q @ n_x + b
        y = H @ x + chol_R @ n_y + c
        return x, (x, y)

    _, (true_states, observations) = jax.lax.scan(body, x0, (normals_x, normals_y))

    return jnp.insert(true_states, 0, x0), observations
