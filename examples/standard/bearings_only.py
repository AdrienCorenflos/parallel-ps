"""
Bearings-only smoothing experiment design
"""
# IMPORTS
import jax.numpy as jnp
from matplotlib import pyplot as plt

from examples.models.bearings_only import make_model, plot_bearings
from parsmooth import MVNSqrt, MVNStandard
from parsmooth.linearization import cubature
from parsmooth.methods import iterated_smoothing, filtering

# CONFIG

s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1., 1.])  # Second sensor location
r = 0.5  # Observation noise (stddev)
dt = 0.01  # discretisation time step
x0 = jnp.array([0.1, 0.2, 1, 0])  # initial true location
qc = 0.01  # discretisation noise
qw = 0.1  # discretisation noise

T = 1_000  # number of observations

np_seed = 42
jax_seed = 666

# GET MODEL

(ts, xs, ys, kalman_observation_model, kalman_transition_model, initialisation_model, transition_kernel,
 observation_potential, m0,
 chol_P0) = make_model(x0, qc, qw, r, dt, s1, s2, T, np_seed)

fixed_point_ICKS = iterated_smoothing(ys, MVNStandard(m0, chol_P0 @ chol_P0.T), kalman_transition_model,
                                      kalman_observation_model, cubature, parallel=True,
                                      criterion=lambda i, *_: i < 1)

filter_CKS = filtering(ys, MVNStandard(m0, chol_P0 @ chol_P0.T), kalman_transition_model, kalman_observation_model,
                       cubature, parallel=False)

# PLOT THE DESIRED OUTCOME UNDER LINEARIZATION
plot_bearings([xs, fixed_point_ICKS.mean, filter_CKS.mean],
              ["True", "ICKS", "CKF"],
              s1, s2, figsize=(15, 10), quiver=False)

plt.show()
