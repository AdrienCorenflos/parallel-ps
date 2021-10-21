"""
Theta-logistic smoothing experiment design
"""

import chex
import jax.numpy as jnp

from parallel_ps.base import BivariatePotentialModel, PyTree, UnivariatePotentialModel
from tests.lgssm import mvn_loglikelihood


class InitialModel(UnivariatePotentialModel):
    def __init__(self, m0, chol_P0):
        super(UnivariatePotentialModel, self).__init__((m0, chol_P0),
                                                       (False, False))

    def log_potential(self, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        m0, chol_P0 = parameter
        return mvn_loglikelihood(x_t, m0, chol_P0)


class TransitionKernel(BivariatePotentialModel):
    def __init__(self, tau_0, tau_1, tau_2, chol_Q):
        super(TransitionKernel, self).__init__((tau_0, tau_1, tau_2, chol_Q),
                                               (False,) * 4)

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        tau_0, tau_1, tau_2, chol_Q = parameter
        mean_x_t = transition_function(x_t_1, tau_0, tau_1, tau_2)
        return mvn_loglikelihood(x_t, mean_x_t, chol_Q)


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
