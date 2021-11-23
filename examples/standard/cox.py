import os
import time
from functools import reduce, partial
from operator import mul

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.contrib.itertools import product

from examples.models.cox import get_data, TransitionKernel, ObservationKernel, InitObservationPotential, InitialModel
from parallel_ps.base import NullPotentialModel, DensityModel, PyTree
from parallel_ps.core.resampling import systematic
from parallel_ps.parallel_smoother import smoothing
from parallel_ps.sequential import smoothing as ffbs_smoothing
# CONFIG
from parallel_ps.utils import mvn_loglikelihood

# from itertools import product

backend = "gpu"
n_smoothers = 100  # number of  times we run the smoother on each dataset

# SSM Config
mu = 0.
rho = 0.9
sigma = 0.5

Ts = np.logspace(5, 14, 9, base=2, dtype=int) - 1
Ns = [25, 50, 75, 100]
use_FFBS = False
use_conditional_proposal = True

# data seed
np.random.seed(0)
JAX_KEYS = jax.random.split(jax.random.PRNGKey(42), n_smoothers)

transition_model = TransitionKernel(mu, sigma, rho)
initial_model = InitialModel(mu, sigma, rho)


class StationaryProposalModel(DensityModel):
    def __init__(self, T):
        chol = sigma / (1. - rho ** 2) ** 0.5
        super(StationaryProposalModel, self).__init__(chol, False, T)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        chol = self.parameters
        eps = jax.random.normal(key, (self.T, N, 1))
        return mu + chol * eps

    def log_potential(self, particle: chex.ArrayTree, chol: PyTree) -> jnp.ndarray:
        return mvn_loglikelihood(particle, mu, chol, is_diag=True)


class ApproximatedConditionalStationaryProposalModel(DensityModel):
    def __init__(self, ys):
        T = ys.shape[0]
        sigma_2 = sigma ** 2 / (1. - rho ** 2)
        super(ApproximatedConditionalStationaryProposalModel, self).__init__((sigma_2, ys), (False, True), T)

    @staticmethod
    def _make_one(sigma_2, y):
        exp_mu = jnp.exp(mu)
        exp_2mu = jnp.exp(2 * mu)
        r_2 = (sigma_2 * exp_2mu + exp_mu)
        mean = mu + jnp.exp(mu) * (y - exp_mu) * sigma_2 / r_2
        chol = (sigma_2 - exp_2mu / r_2) ** 0.5
        return mean, chol

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        sigma_2, ys = self.parameters
        means, chols = jax.vmap(self._make_one, [None, 0])(sigma_2, ys)
        eps = jax.random.normal(key, (self.T, N, 1))
        return means[:, None, None] + chols[..., None] * eps

    def log_potential(self, particle: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        sigma_2, y = parameters
        mean, chol = self._make_one(sigma_2, y)
        return mvn_loglikelihood(particle, mean, chol, is_diag=True)


def phi(x):
    x = x[..., 0]
    T = x.shape[0]
    x0 = x[0]

    x_t_p_1 = x[1:]
    x_t = x[:-1]

    sig_2 = sigma ** 2
    sig_4 = sigma ** 4

    res = -0.5 * (T + 1) / sig_2 + 0.5 * (1 - rho ** 2) * (x0 - mu) ** 2 / sig_4
    res = res + 0.5 * jnp.sum((x_t_p_1 - mu - rho * (x_t - mu)) ** 2) / sig_4
    return res


def experiment(T, N):
    xs, ys = get_data(mu, rho, sigma, T)
    observation_model = ObservationKernel(ys[1:])
    initial_observation_model = InitObservationPotential(ys[0])

    @jax.jit
    def one_smoother(k):
        if use_FFBS:
            (_, trajectories), _ = ffbs_smoothing(T, k, transition_model, observation_model, initial_model,
                                                  initial_observation_model, N=N, M=N)
            return trajectories
        else:
            if use_conditional_proposal:
                proposal_model = ApproximatedConditionalStationaryProposalModel(ys)
            else:
                proposal_model = StationaryProposalModel(T)
            ps_result = smoothing(k, proposal_model, proposal_model,
                                  transition_model, observation_model, initial_model,
                                  NullPotentialModel(), systematic, N=N)

            return ps_result.trajectories

    @partial(jax.jit, backend=backend)
    def compute_score_func(k):
        trajectories = one_smoother(k)
        trajectory_score = jax.vmap(phi, in_axes=[1])(trajectories)
        return jnp.mean(trajectory_score)

    runtimes = np.empty((len(JAX_KEYS),))
    scores = np.empty((len(JAX_KEYS),))
    score_mean = compute_score_func(JAX_KEYS[0])
    score_mean.block_until_ready()
    for i, jax_key in enumerate(JAX_KEYS):
        tic = time.time()
        score_mean = compute_score_func(jax_key)
        score_mean.block_until_ready()
        toc = time.time()
        runtimes[i] = toc - tic
        scores[i] = score_mean

    return np.mean(runtimes), scores


shape = (len(Ts), len(Ns))
runtime_means = np.empty(shape)
batch_scores = np.empty(shape + (n_smoothers,))

indices = np.recarray(shape,
                      dtype=[("T", int), ("N", int)])

for (m, T), (n, N) in product(
        *map(enumerate, [Ts, Ns]), total=reduce(mul, shape)):
    runtime, batch_score = experiment(T, N)

    runtime_means[m, n] = runtime
    batch_scores[m, n, :] = batch_score
    indices[m, n]["T"] = T
    indices[m, n]["N"] = N

os.makedirs("./output", exist_ok=True)
np.savez(f"./output/cox-{use_FFBS}-{use_conditional_proposal}-{backend}",
         indices=indices,
         runtime_means=runtime_means,
         batch_scores=batch_scores)