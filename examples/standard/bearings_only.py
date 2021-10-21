"""
Bearings-only smoothing experiment design
"""
# IMPORTS
import time

import chex
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from examples.models.bearings_only import make_model, plot_bearings
from parallel_ps.base import DensityModel, PyTree, UnivariatePotentialModel, NullPotentialModel
from parallel_ps.core.resampling import systematic
from parallel_ps.ffbs_smoother import smoothing as ffbs
from parallel_ps.parallel_smoother import smoothing as particle_smoothing
from parsmooth import MVNSqrt
from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing
from tests.lgssm import mvn_loglikelihood

# CONFIG
### Particle smoother config
N = 100  # Number of particles
B = 25  # Number of smoothers run for stats

### Model config
s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1., 1.])  # Second sensor location
r = 0.1  # Observation noise (stddev)
dt = 0.01  # discretisation time step
x0 = jnp.array([0.1, 0.2, 1., 0.])  # initial true location
qc = 0.5  # discretisation noise
qw = 5.  # discretisation noise

T = 1_000  # number of observations

### Other config
backend = "gpu"
np_seed = 42  # Same seed as in paper with Fatemeh
jax_seed = 0

np.random.seed(np_seed)
jax_key = jax.random.PRNGKey(jax_seed)

# GET MODEL

(ts, xs, ys, kalman_observation_model, kalman_transition_model, initialisation_model, transition_kernel,
 observation_potential, m0,
 chol_P0) = make_model(x0, qc, qw, r, dt, s1, s2, T)

jitted_iterated_smoother = jax.jit(
    lambda: iterated_smoothing(ys, MVNSqrt(m0, chol_P0), kalman_transition_model, kalman_observation_model,
                               extended, parallel=True, criterion=lambda i, *_: i < 20), backend=backend)

fixed_point_ICKS = jitted_iterated_smoother()
print("kalman smoother has converged")


# DEFINE nu_t and q_t

class NutModel(UnivariatePotentialModel):
    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mean, chol = parameter
        return mvn_loglikelihood(particles, mean, chol)


class QtModel(DensityModel):
    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mean, chol = parameter
        return mvn_loglikelihood(particles, mean, chol)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        means, chols = self.parameters
        normals = jax.random.normal(key, (self.T, N, means.shape[-1]))
        return means[:, None, :] + jnp.einsum("...ij,...kj->...ki", chols, normals)


### INSTANTIATE
nu_t = NutModel(fixed_point_ICKS, MVNSqrt(True, True))
q_t = QtModel(fixed_point_ICKS, MVNSqrt(True, True), T=T + 1)

### RUN

ps_from_key = jax.jit(lambda key: particle_smoothing(key, q_t, nu_t, transition_kernel, observation_potential,
                                                     initialisation_model, NullPotentialModel(), systematic, N),
                      backend=backend)

compile_smoother = ps_from_key(jax_key)
compile_smoother.trajectories.block_until_ready()
print("parallel particle smoother compiled")
tic = time.time()
for _ in range(B):
    particle_smoothing_results = ps_from_key(jax_key)
    particle_smoothing_results.trajectories.block_until_ready()
toc = time.time()

print("parallel particle smoother time:", (toc - tic) / B)

ffbs_from_key = jax.jit(lambda key: ffbs(key, q_t, transition_kernel, observation_potential,
                                         initialisation_model, NullPotentialModel(), N), backend=backend)

ffbs_compile, _ = ffbs_from_key(jax_key)
ffbs_compile.block_until_ready()

print("FFBS compiled")

tic = time.time()
for _ in range(B):
    ffbs_results, ffbs_sampled_indices = ffbs_from_key(jax_key)
    ffbs_results.block_until_ready()
toc = time.time()

print("FFBS time:", (toc - tic) / B)


n_unique_ps = np.mean([len(np.unique(particle_smoothing_results.origins[t])) for t in range(T + 1)])
n_unique_ffbs = np.mean([len(np.unique(ffbs_sampled_indices[t])) for t in range(T + 1)])

print("PS unique origins:", n_unique_ps)
print("FFBS unique origins:", n_unique_ffbs)


plot_bearings([xs, fixed_point_ICKS.mean, particle_smoothing_results.trajectories.mean(1), ffbs_results.mean(1)],
              ["True", "ICKS", "PS-mean", "FF-BS-mean"],
              s1, s2, figsize=(15, 10), quiver=False)

plt.show()