"""
Bearings-only smoothing experiment design
"""
# IMPORTS
import chex
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from examples.models.bearings_only import make_model, plot_bearings
from parallel_ps.base import DensityModel, PyTree, UnivariatePotentialModel, NullPotentialModel
from parallel_ps.core.resampling import stratified, systematic
from parallel_ps.smoother import smoothing as particle_smoothing
from parsmooth import MVNSqrt
from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing, sampling, filtering
from tests.lgssm import mvn_loglikelihood

# CONFIG
### Particle smoother config
N = 100  # Number of particles
B = 5  # Number of smoothers ran for stats

### Model config
s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1., 1.])  # Second sensor location
r = 0.5  # Observation noise (stddev)
dt = 0.01  # discretisation time step
x0 = jnp.array([0.1, 0.2, 1., 0.])  # initial true location
qc = 0.01  # discretisation noise
qw = 0.1  # discretisation noise

T = 1_000  # number of observations

### Other config
np_seed = 42  # Same seed as in paper with Fatemeh
jax_seed = 123456

np.random.seed(np_seed)
jax_key = jax.random.PRNGKey(jax_seed)

# GET MODEL

(ts, xs, ys, kalman_observation_model, kalman_transition_model, initialisation_model, transition_kernel,
 observation_potential, m0,
 chol_P0) = make_model(x0, qc, qw, r, dt, s1, s2, T)

fixed_point_ICKS = iterated_smoothing(ys, MVNSqrt(m0, chol_P0), kalman_transition_model, kalman_observation_model,
                                      extended, parallel=True, criterion=lambda i, *_: i < 100)

filtering_trajectory = filtering(ys, MVNSqrt(m0, chol_P0), kalman_transition_model, kalman_observation_model,
                                 extended, fixed_point_ICKS, parallel=True)


# DEFINE nu_t and q_t

class NutModel(UnivariatePotentialModel):
    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        return mvn_loglikelihood(particles, *parameter)


class QtModel(DensityModel):
    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        return mvn_loglikelihood(particles, *parameter)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        return sampling(key, N, kalman_transition_model, filtering_trajectory, extended, self.parameters, parallel=True)


### INSTANTIATE
nu_t = NutModel(fixed_point_ICKS, MVNSqrt(True, True))
q_t = QtModel(fixed_point_ICKS, MVNSqrt(True, True), T=T + 1)

ps_from_key = jax.jit(lambda key: particle_smoothing(key, q_t, nu_t, transition_kernel,
                                                     observation_potential, initialisation_model,
                                                     NullPotentialModel(), systematic, N, coupled=True))

particle_smoothing_result = ps_from_key(jax_key)

print()
print(particle_smoothing_result.origins)

n_unique = np.mean([len(np.unique(particle_smoothing_result.origins[t])) for t in range(T + 1)])

print(n_unique)


# n_samples = 5
# samples = q_t.sample(jax_keys[0], n_samples)

plot_bearings([xs, fixed_point_ICKS.mean, particle_smoothing_result.trajectories[:].mean(1)],
              ["True", "ICKS", "PS-mean"],
              s1, s2, figsize=(15, 10), quiver=False)

plt.show()
