"""
Theta-logistic smoothing experiment design
"""

import chex
import jax
import jax.numpy as jnp

from parallel_ps.base import BivariatePotentialModel, PyTree, UnivariatePotentialModel, ConditionalDensityModel, \
    DensityModel
from parallel_ps.utils import mvn_loglikelihood


class InitialModel(DensityModel):
    def __init__(self, m0, chol_P0):
        super(DensityModel, self).__init__((m0, chol_P0),
                                           (False, False))

    def log_potential(self, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        m0, chol_P0 = parameter
        return mvn_loglikelihood(x_t, m0, chol_P0)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        m0, chol_P0 = self.parameters
        eps = jax.random.normal(key, (N, 1))
        return m0 + chol_P0 * eps


class TransitionKernel(ConditionalDensityModel):

    def __init__(self, tau_0, tau_1, tau_2, chol_Q):
        super(TransitionKernel, self).__init__((tau_0, tau_1, tau_2, chol_Q),
                                               (False,) * 4)

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        tau_0, tau_1, tau_2, chol_Q = parameter
        mean_x_t = transition_function(x_t_1, tau_0, tau_1, tau_2)
        return mvn_loglikelihood(x_t, mean_x_t, chol_Q)

    def sample(self, key: chex.PRNGKey, x_t_1: chex.ArrayTree, parameter: PyTree) -> chex.ArrayTree:
        tau_0, tau_1, tau_2, chol_Q = parameter
        N = x_t_1.shape[0]
        vmapped_transition_function = jax.vmap(transition_function, [0, None, None, None])
        mean_x_t = vmapped_transition_function(x_t_1, tau_0, tau_1, tau_2)
        eps = jax.random.normal(key, (N, 1))
        return mean_x_t + chol_Q * eps


class LocallyOptimalTransitionKernel(ConditionalDensityModel):

    def __init__(self, tau_0, tau_1, tau_2, chol_Q, chol_R, ys):
        super(LocallyOptimalTransitionKernel, self).__init__((tau_0, tau_1, tau_2, chol_Q, chol_R, ys),
                                                             (False,) * 5 + (True,))

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        tau_0, tau_1, tau_2, chol_Q, chol_R, yt = parameter
        mean_x_t = transition_function(x_t_1, tau_0, tau_1, tau_2)

        r = chol_R[0, 0]
        q = chol_Q[0, 0]

        post_cov_xt = 1. / (1. / r ** 2 + 1. / q ** 2)
        post_mean_xt = post_cov_xt * (mean_x_t / q ** 2 + yt / r ** 2)

        return mvn_loglikelihood(x_t, post_mean_xt, jnp.atleast_2d(post_cov_xt ** 0.5))

    def sample(self, key: chex.PRNGKey, x_t_1: chex.ArrayTree, parameter: PyTree) -> chex.ArrayTree:
        tau_0, tau_1, tau_2, chol_Q, chol_R, yt = parameter
        N = x_t_1.shape[0]
        vmapped_transition_function = jax.vmap(transition_function, [0, None, None, None])
        mean_x_t = vmapped_transition_function(x_t_1, tau_0, tau_1, tau_2)
        r = chol_R[0, 0]
        q = chol_Q[0, 0]

        post_cov_xt = 1. / (1. / r ** 2 + 1. / q ** 2)
        post_mean_xt = post_cov_xt * (mean_x_t / q ** 2 + yt / r ** 2)
        eps = jax.random.normal(key, (N, 1))

        return post_mean_xt + jnp.sqrt(post_cov_xt) * eps


class LocallyOptimalObservationKernel(BivariatePotentialModel):

    def __init__(self, tau_0, tau_1, tau_2, chol_Q, chol_R, ys):
        super(LocallyOptimalObservationKernel, self).__init__((tau_0, tau_1, tau_2, chol_Q, chol_R, ys),
                                                              (False,) * 5 + (True,))

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        tau_0, tau_1, tau_2, chol_Q, chol_R, y_t = parameter
        mean_x_t = transition_function(x_t_1, tau_0, tau_1, tau_2)

        r = chol_R[0, 0]
        q = chol_Q[0, 0]

        post_cov_xt = 1. / (1. / r ** 2 + 1. / q ** 2)
        post_mean_xt = post_cov_xt * (mean_x_t / q ** 2 + y_t / r ** 2)

        proposal_loglik = mvn_loglikelihood(x_t, post_mean_xt, jnp.atleast_2d(post_cov_xt ** 0.5))
        transition_loglik = mvn_loglikelihood(x_t, transition_function(x_t_1, tau_0, tau_1, tau_2), chol_Q)
        observation_loglik = mvn_loglikelihood(y_t, x_t, chol_R)
        return transition_loglik + observation_loglik - proposal_loglik


class ObservationPotential(BivariatePotentialModel):
    def __init__(self, chol_R, y_t):
        super(ObservationPotential, self).__init__((chol_R, y_t),
                                                   (False, True))

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        cholR, y_t = parameter
        mean_y_t = observation_function(x_t)
        return mvn_loglikelihood(y_t, mean_y_t, cholR)


class InitObservationPotential(UnivariatePotentialModel):
    def __init__(self, chol_R, y_t):
        super(UnivariatePotentialModel, self).__init__((chol_R, y_t),
                                                       (False, False))

    def log_potential(self, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        cholR, y_t = parameter
        mean_y_t = observation_function(x_t)
        return mvn_loglikelihood(y_t, mean_y_t, cholR)


def transition_function(x, tau_0, tau_1, tau_2):
    """ Deterministic transition function used in the state space model
    Parameters
    ----------
    x: array_like
        The current state
    tau_0, tau_1, tau_2: float
        The parameters of the model

    Returns
    -------
    out: array_like
        The transitioned state
    """
    return x + tau_0 - tau_1 * jnp.exp(tau_2 * x)


def observation_function(x):
    """
    Returns the observed value

    Parameters
    ----------
    x: array_like
        The current state
    Returns
    -------
    y: array_like
        The observed angles, the first component is the angle w.r.t. the first sensor, the second w.r.t the second.
    """
    return x
