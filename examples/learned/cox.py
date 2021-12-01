import os
import time
from functools import reduce, partial
from operator import mul

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.optimizers import adam
from tqdm.contrib.itertools import product

from examples.models.cox import get_data, TransitionKernel, ObservationKernel, InitObservationPotential, InitialModel
from parallel_ps.base import DensityModel, PyTree
from parallel_ps.core.resampling import systematic
from parallel_ps.parallel_smoother import smoothing
# CONFIG
from parallel_ps.utils import mvn_loglikelihood

# from itertools import product

backend = "gpu"
n_smoothers = 100  # number of times we run the smoother

# SSM Config
mu = 0.
rho = 0.9
sigma = 0.5

Ts = np.logspace(6, 14, 8, base=2, dtype=int)
Ns = [25, 50, 75, 100, 125, 150, 175, 200]

# data seed
np.random.seed(1234)
JAX_KEYS = jax.random.split(jax.random.PRNGKey(42), n_smoothers)

transition_model = TransitionKernel(mu, sigma, rho)
initial_model = InitialModel(mu, sigma, rho)

B = 4  # number of parallel smoothers to learn the proposal
n_iter = 10
learning_rate = 1e-3
optimizer = adam(learning_rate)
opt_key = jax.random.PRNGKey(31415926)


class ProposalModel(DensityModel):
    def __init__(self, means, chols, T):
        super(ProposalModel, self).__init__((means, chols), (True, True), T)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        means, chols = self.parameters
        eps = jax.random.normal(key, (self.T, N, 1))
        return means[:, None, None] + chols[:, None, None] * eps

    def log_potential(self, particle: chex.ArrayTree, params: PyTree) -> jnp.ndarray:
        mean, chol = params
        return mvn_loglikelihood(particle, mean, chol, is_diag=True)


def phi(x):
    x = x[..., 0]
    T = x.shape[0]
    x0 = x[0]

    x_t_p_1 = x[1:]
    x_t = x[:-1]

    sig_2 = sigma ** 2
    sig_4 = sig_2 ** 2

    res = -0.5 * T / sig_2 + 0.5 * (1 - rho ** 2) * (x0 - mu) ** 2 / sig_4
    res = res + 0.5 * jnp.sum((x_t_p_1 - mu - rho * (x_t - mu)) ** 2) / sig_4
    return res


def experiment(T, N):
    xs, ys = get_data(mu, rho, sigma, T)
    observation_model = ObservationKernel(ys[1:])
    initial_observation_model = InitObservationPotential(ys[0])

    @jax.jit
    def loss_one(qt_mean, qt_log_chol, nu_t_mean, nu_t_log_chol, k):
        proposal_model = ProposalModel(qt_mean, jnp.exp(qt_log_chol), T)
        nut_model = ProposalModel(nu_t_mean, jnp.exp(nu_t_log_chol), T)
        ps = lambda k: smoothing(k, proposal_model, nut_model,
                                 transition_model, observation_model, initial_model,
                                 initial_observation_model, systematic, N=N)
        ps_result = ps(k)
        phi_res = jax.vmap(phi, in_axes=[1])(ps_result.trajectories)
        return jnp.mean(phi_res)

    @jax.jit
    def learn_param():
        init_mu, init_chol = mu, sigma / (1 - rho ** 2) ** 0.5

        init_mu = init_mu * jnp.ones((T,))
        init_chol_log = jnp.log(init_chol * jnp.ones((T,)))

        opt_state_init = optimizer.init_fn((init_mu, init_chol_log, init_mu, init_chol_log))

        def loss(qt_mean, qt_log_chol, nu_t_mean, nu_t_log_chol, k):
            keys = jax.random.split(k, B)
            fun = lambda k_: loss_one(qt_mean, qt_log_chol, nu_t_mean, nu_t_log_chol, k_)
            phi_res = jax.vmap(fun)(keys)
            phi_var = jnp.var(phi_res, 0)
            return phi_var

        def loop(opt_state, inputs):
            i, k = inputs
            _, grads = jax.value_and_grad(loss, [0, 1, 2, 3])(*optimizer.params_fn(opt_state), k)
            grads = jax.tree_map(lambda z: jnp.nan_to_num(z), grads)
            return optimizer.update_fn(i, grads, opt_state), None

        final_opt_state, _ = jax.lax.scan(loop, opt_state_init,
                                          (jnp.arange(n_iter), jax.random.split(opt_key, n_iter)))
        return optimizer.params_fn(final_opt_state)

    tic = time.time()
    learned_params = learn_param()
    learned_params[0].block_until_ready()
    learning_time = time.time() - tic

    runtimes = np.empty((len(JAX_KEYS),))
    scores = np.empty((len(JAX_KEYS),))
    score_mean = loss_one(*learned_params, JAX_KEYS[0])
    score_mean.block_until_ready()

    for i, jax_key in enumerate(JAX_KEYS):
        tic = time.time()
        score_mean = loss_one(*learned_params, jax_key)
        score_mean.block_until_ready()
        toc = time.time()
        runtimes[i] = toc - tic
        scores[i] = score_mean

    return np.mean(runtimes), scores, learning_time


shape = (len(Ts), len(Ns))
runtime_means = np.empty(shape)
learning_times = np.empty(shape)
batch_scores = np.empty(shape + (n_smoothers,))

indices = np.recarray(shape,
                      dtype=[("T", int), ("N", int)])

for (m, T), (n, N) in product(
        *map(enumerate, [Ts, Ns]), total=reduce(mul, shape)):
    runtime, batch_score, learning_time = experiment(T, N)
    learning_times[m, n] = learning_time
    runtime_means[m, n] = runtime
    batch_scores[m, n, :] = batch_score
    indices[m, n]["T"] = T
    indices[m, n]["N"] = N

os.makedirs("./output", exist_ok=True)
np.savez(f"./output/cox-learned-{backend}",
         indices=indices,
         runtime_means=runtime_means,
         batch_scores=batch_scores,
         learning_times=learning_times)
