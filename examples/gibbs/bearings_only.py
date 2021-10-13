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
from parallel_ps.core.resampling import multinomial
from parallel_ps.smoother import smoothing as particle_smoothing
from parsmooth import MVNSqrt
from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing, sampling, filtering
from tests.lgssm import mvn_loglikelihood

# CONFIG
### Particle smoother config
N = 250  # Number of particles
B = 100  # Number of smoothers ran for stats

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

cps_from_key = jax.jit(lambda key, traj: particle_smoothing(key, q_t, nu_t, transition_kernel, observation_potential,
                                                            initialisation_model, NullPotentialModel(), multinomial, N,
                                                            conditional_trajectory=traj, coupled=True)[0])


### DEFINE GIBBS ROUTINE

def gibbs_routine(rng_key, n_iter, init_traj):
    def count_rejuvenate(origins):
        return np.sum(origins != 0, axis=1)

    def body(curr_traj, inps):
        from jax.experimental.host_callback import id_tap

        i, curr_key = inps
        sample_key, select_key = jax.random.split(curr_key)
        samples = id_tap(lambda j: print(f"iteration {j}/{n_iter}", end="\r"), i,
                         result=cps_from_key(sample_key, curr_traj))
        rejuvenated = count_rejuvenate(samples.origins)
        next_traj = samples.trajectories[:, jax.random.randint(select_key, (), 0, N - 1)]
        return next_traj, (next_traj, rejuvenated)

    keys = jax.random.split(rng_key, n_iter)
    _, (gibbs_samples, rejuvenated) = jax.lax.scan(body, init_traj, (jnp.arange(n_iter), keys))
    return jnp.swapaxes(gibbs_samples, 0, 1), rejuvenated


init_key, gibbs_key = jax.random.split(jax_key)
init_traj = sampling(init_key, 1, kalman_transition_model, filtering_trajectory, extended, fixed_point_ICKS, True)

particle_smoothing_result, rejuvenation_logs = gibbs_routine(gibbs_key, B, init_traj[:, 0])
print()
plt.plot(np.arange(T + 1), rejuvenation_logs.mean(0))
plt.show()

plot_bearings([xs, fixed_point_ICKS.mean, particle_smoothing_result.mean(1)],
              ["True", "ICKS", "Gibbs"],
              s1, s2, figsize=(15, 10), quiver=False)

plt.show()
