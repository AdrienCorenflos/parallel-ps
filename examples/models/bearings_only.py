"""
Bearings-only smoothing experiment design
"""

from functools import partial

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
from jax import lax, jit

from parallel_ps.base import BivariatePotentialModel, PyTree, UnivariatePotentialModel
from parsmooth import FunctionalModel, MVNSqrt
from tests.lgssm import mvn_loglikelihood


class BearingsInitialModel(UnivariatePotentialModel):
    def __init__(self, m0, chol_P0):
        super(UnivariatePotentialModel, self).__init__((m0, chol_P0), (False, False))

    def log_potential(self, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        m0, chol_P0 = parameter
        return mvn_loglikelihood(x_t, m0, chol_P0)


class BearingsTransitionKernel(BivariatePotentialModel):
    def __init__(self, transition_function, chol_Q):
        super(BearingsTransitionKernel, self).__init__(chol_Q, False)
        self._transition_function = transition_function

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        cholQ = parameter
        mean_x_t = self._transition_function(x_t_1)
        return mvn_loglikelihood(x_t, mean_x_t, cholQ)


class BearingsObservationPotential(BivariatePotentialModel):
    def __init__(self, observation_function, chol_R, y_t):
        super(BearingsObservationPotential, self).__init__((chol_R, y_t), (False, True))
        self._observation_function = observation_function

    def log_potential(self, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        cholR, y_t = parameter
        mean_y_t = self._observation_function(x_t)
        return mvn_loglikelihood(y_t, mean_y_t, cholR)


def _transition_function(x, dt):
    """ Deterministic transition function used in the state space model
    Parameters
    ----------
    x: array_like
        The current state
    dt: float
        Time step between observations
    Returns
    -------
    out: array_like
        The transitioned state
    """
    w = x[-1]
    predicate = jnp.abs(w) < 1e-6

    coswt = jnp.cos(w * dt)
    sinwt = jnp.sin(w * dt)

    def true_fun(_):
        return coswt, 0., sinwt, dt

    def false_fun(_):
        coswto = coswt - 1
        return coswt, coswto / w, sinwt, sinwt / w

    coswt, coswtopw, sinwt, sinwtpw = lax.cond(predicate, true_fun, false_fun, None)

    F = jnp.array([[1, 0, sinwtpw, -coswtopw, 0],
                   [0, 1, coswtopw, sinwtpw, 0],
                   [0, 0, coswt, sinwt, 0],
                   [0, 0, -sinwt, coswt, 0],
                   [0, 0, 0, 0, 1]])

    return F @ x


def _observation_function(x, s1, s2):
    """
    Returns the observed angles as function of the state and the sensors locations
    Parameters
    ----------
    x: array_like
        The current state
    s1: array_like
        The first sensor location
    s2: array_like
        The second sensor location
    Returns
    -------
    y: array_like
        The observed angles, the first component is the angle w.r.t. the first sensor, the second w.r.t the second.
    """
    return jnp.array([jnp.arctan2(x[1] - s1[1], x[0] - s1[0]),
                      jnp.arctan2(x[1] - s2[1], x[0] - s2[0])])


def make_model(x0, qc, qw, r, dt, s1, s2, T):
    """ Discretizes the model with continuous transition noise qc, for step-size dt.
    The model is described in "Multitarget-multisensor tracking: principles and techniques" by
    Bar-Shalom, Yaakov and Li, Xiao-Rong
    Parameters
    ----------
    x0: jnp.ndarray
        Initial true location
    qc: float
        Transition covariance of the continuous SSM
    qw: float
        Transition covariance of the continuous SSM
    r: float
        Observation error standard deviation
    dt: float
        Discretization time step
    s1: array_like
        The location of the first sensor
    s2: array_like
        The location of the second sensor
    T:  number of time steps for the model
    """

    Q = jnp.array([[qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0, 0],
                   [0, qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0],
                   [qc * dt ** 2 / 2, 0, qc * dt, 0, 0],
                   [0, qc * dt ** 2 / 2, 0, qc * dt, 0],
                   [0, 0, 0, 0, dt * qw]])

    chol_Q = jnp.linalg.cholesky(Q)
    chol_R = r * jnp.eye(2)

    observation_function = partial(_observation_function, s1=s1, s2=s2)
    transition_function = partial(_transition_function, dt=dt)

    kalman_observation_model = FunctionalModel(jit(lambda x, r_val: observation_function(x) + r_val),
                                               MVNSqrt(jnp.zeros((2,)), chol_R))
    kalman_transition_model = FunctionalModel(jit(lambda x, q: transition_function(x) + q),
                                              MVNSqrt(jnp.zeros((5,)), chol_Q))

    m0 = jnp.array([-1., -1., 0., 0., 0.])
    P0 = chol_P0 = jnp.eye(5)
    ts, xs, ys = get_data(x0, dt, r, T, s1, s2)

    initialisation_model = BearingsInitialModel(m0, chol_P0)
    transition_kernel = BearingsTransitionKernel(transition_function, chol_Q)
    observation_potential = BearingsObservationPotential(observation_function, chol_R, jnp.asarray(ys))

    return (ts, xs, ys, kalman_observation_model, kalman_transition_model, initialisation_model, transition_kernel,
            observation_potential, m0, P0)


def _get_data(x, dt, a_s, s1, s2, r, normals, observations, true_states):
    for i, a in enumerate(a_s):
        F = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, a],
                      [0, 0, -a, 0]])
        x = linalg.expm(F * dt) @ x
        y1 = np.arctan2(x[1] - s1[1], x[0] - s1[0]) + r * normals[i, 0]
        y2 = np.arctan2(x[1] - s2[1], x[0] - s2[0]) + r * normals[i, 1]

        observations[i] = [y1, y2]
        observations[i] = [y1, y2]
        true_states[i] = np.concatenate((x, np.array([a])))


def get_data(x0, dt, r, T, s1, s2, q=10.):
    """
    Parameters
    ----------
    x0: array_like
        true initial state
    dt: float
        time step for observations
    r: float
        observation model standard deviation
    T: int
        number of time steps
    s1: array_like
        The location of the first sensor
    s2: array_like
        The location of the second sensor
    q: float
        noise of the angular momentum

    Returns
    -------
    ts: array_like
        array of time steps
    true_states: array_like
        array of true states
    observations: array_like
        array of observations
    """

    a_s = 1 + q * dt * np.cumsum(np.random.randn(T))
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)

    x = np.copy(x0)
    observations = np.empty((T, 2))
    true_states = np.empty((T, 5))
    ts = np.linspace(dt, (T + 1) * dt, T)

    normals = np.random.randn(T, 2)

    _get_data(x, dt, a_s, s1, s2, r, normals, observations, true_states)
    return ts, true_states, observations


def plot_bearings(states, labels, s1, s2, figsize=(10, 10), quiver=False):
    """
    Parameters
    ----------
    states: list of array_like
        list of states to plot
    labels: list of str
        list of lables for the states
    s1: array_like
        first sensor
    s2: array_like
        second sensor
    figsize: tuple of int
        figure size in inches
    quiver: bool
        show the velocity field

    """

    fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(states, list):
        states = [states]

    if not isinstance(labels, list):
        labels = [labels]

    for label, state in zip(labels, states):
        ax.plot(*state[:, :2].T, linestyle='--', label=label, alpha=0.75)
        if quiver:
            ax.quiver(*state[::10].T, units='xy', scale=4, width=0.01)
    ax.scatter(*s1, marker="o", s=200, label="Sensor 1", color='k')
    ax.scatter(*s2, marker="x", s=200, label="Sensor 2", color='k')

    ax.legend(loc="lower left")