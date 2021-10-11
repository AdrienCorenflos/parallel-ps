"""
Bearings-only smoothing experiment design
"""
# IMPORTS
import chex
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from jax.experimental.optimizers import adam
from matplotlib import pyplot as plt

from examples.models.bearings_only import make_model, plot_bearings
from parallel_ps.base import DensityModel, PyTree, UnivariatePotentialModel, NullPotentialModel
from parallel_ps.core.resampling import stratified
from parallel_ps.smoother import smoothing as particle_smoothing
from parsmooth import MVNSqrt
from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing, sampling, filtering
from tests.lgssm import mvn_logprob_fn

# CONFIG
### Particle smoother config
N = 25  # Number of particles
B = 3  # Number of smoothers ran for stats

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
step_size = 1e-4
n_iter = 1_000

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
        return mvn_logprob_fn(particles, *parameter)


class QtModel(DensityModel):
    def log_potential(self, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        return mvn_logprob_fn(particles, *parameter)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        return sampling(key, N, kalman_transition_model, filtering_trajectory, extended, self.parameters, parallel=True)


### INSTANTIATE
q_t = QtModel(fixed_point_ICKS, MVNSqrt(True, True), T=T + 1)


def loss_function(key, nut_params):
    nu_t = NutModel(nut_params, MVNSqrt(True, True))

    @jax.vmap
    def loss_one(k):
        samples = particle_smoothing(k, q_t, nu_t, transition_kernel, observation_potential,
                                     initialisation_model, NullPotentialModel(), stratified, N)
        return -samples.ells[-1] / T

    keys = jax.random.split(key, B)
    return loss_one(keys).mean()


opt_init, opt_update, opt_get_params = adam(step_size)


@jax.jit
def step(rng_key, step, opt_state):
    value, grads = jax.value_and_grad(lambda p: loss_function(rng_key, p))(opt_get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


opt_state = opt_init(fixed_point_ICKS)

pbar = tqdm.trange(n_iter)
keys = jax.random.split(jax_key, n_iter)
for i in pbar:
    value, opt_state = step(keys[i], i, opt_state)
    pbar.set_description("Loss %s" % value)

opt_mean, opt_chol = opt_get_params(opt_state)

opt_nut = NutModel((opt_mean, opt_chol), (True, True))

opt_res = particle_smoothing(jax_key, q_t, opt_nut, transition_kernel, observation_potential,
                             initialisation_model, NullPotentialModel(), stratified, N)

plot_bearings([xs, fixed_point_ICKS.mean, opt_res.trajectories.mean(1)],
              ["True", "ICKS", "Optimised PS"],
              s1, s2, figsize=(15, 10), quiver=False)

plt.show()
