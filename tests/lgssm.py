import chex
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from parallel_ps.base import BivariatePotentialModel, PyTree


def mvn_logprob_fn(x, mean, chol_cov):
    y = jax.scipy.linalg.solve_triangular(chol_cov, x - mean, lower=True)
    return - 0.5 * jnp.sum(y * y, -1)


def _lgssm_log_potential_one(x, y, F, b, cholQ):
    mean = F @ x + b
    return mvn_logprob_fn(y, mean, cholQ)


class LinearGaussianTransitionModel(BivariatePotentialModel):
    parameters: PyTree
    batched: PyTree

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        F, b, cholQ = parameter
        return _lgssm_log_potential_one(x_t_1, x_t, F, b, cholQ)


class LinearGaussianObservationModel(BivariatePotentialModel):
    parameters: PyTree
    batched: PyTree

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        H, c, cholR, y_t = parameter
        return _lgssm_log_potential_one(x_t, y_t, H, c, cholR)


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

    normals_x = np.random.randn(T, dim_x).astype(np.float32)
    normals_y = np.random.randn(T, dim_y).astype(np.float32)
    x = m0 + chol_P0 @ np.random.randn(dim_x)

    observations = np.empty((T, dim_y), dtype=np.float32)
    true_states = np.empty((T + 1, dim_x), dtype=np.float32)
    true_states[0] = x

    for i, (norm_x, norm_y) in enumerate(zip(normals_x, normals_y)):
        x = F @ x + chol_Q @ norm_x + b
        true_states[i+1] = x
        y = H @ x + chol_R @ norm_y + c
        observations[i] = y

    return true_states, observations
