import os
import time
from functools import reduce, partial
from operator import mul

import jax
import jax.numpy as jnp
import numpy as np
from parsmooth.sequential import ekf, eks  # sequential cause we need the log-likelihood just for the experiment.
from parsmooth.utils import MVNormalParameters
from tqdm.contrib.itertools import product

from examples.models.lgssm import get_data, LinearGaussianObservationModel, LinearGaussianTransitionModel
from parallel_ps.base import GaussianDensity, NullPotentialModel
from parallel_ps.core.resampling import systematic
from parallel_ps.ffbs_smoother import smoothing as ffbs_smoothing
from parallel_ps.parallel_smoother import smoothing

# CONFIG

backend = "gpu"
n_smoothers = 50  # number of  times we run the smoother on each dataset

# SSM Config
dim_x = 1
dim_y = 1

Ts = [2 ** k - 1 for k in range(3, 16, 1)]
Ns = [25, 50, 100, 250]
use_FFBS = False

# data seed
np.random.seed(0)
JAX_KEYS = jax.random.split(jax.random.PRNGKey(42), n_smoothers)


def make_model():
    F = 0.9 * np.eye(1)
    H = np.eye(dim_y, dim_x)
    b = np.zeros((dim_x,))
    c = np.zeros((dim_y,))

    chol_Q = 0.1 * np.eye(dim_x)
    chol_R = 0.1 * np.eye(dim_y)

    m0 = np.zeros((dim_x,))
    chol_P0 = 0.1 * np.eye(dim_x)
    return F, H, b, c, chol_Q, chol_R, m0, chol_P0


def experiment(T, N):
    F, H, b, c, chol_Q, chol_R, m0, chol_P0 = make_model()

    Q = chol_Q @ chol_Q.T
    R = chol_R @ chol_R.T
    P0 = chol_P0 @ chol_P0.T

    xs, ys = get_data(m0, chol_P0, F, H, chol_R, chol_Q, b, c, T)

    transition_function = lambda x: F @ x + b
    observation_function = lambda x: H @ x + c

    x0 = MVNormalParameters(m0, P0)
    jitted_ekf = jax.jit(ekf, static_argnums=(2, 4, 6, 7), backend="cpu")
    jitted_eks = jax.jit(eks, static_argnums=(0, 3, 4), backend="cpu")

    kalman_ell, kalman_filtering_solution = jitted_ekf(x0, ys,
                                                       transition_function,
                                                       Q,
                                                       observation_function,
                                                       R,
                                                       None, True)

    kalman_smoothing_solution = jitted_eks(transition_function, Q,
                                           kalman_filtering_solution,
                                           None, True)

    kalman_smoothing_means, kalman_smoothing_covs = kalman_smoothing_solution

    transition_model = LinearGaussianTransitionModel(
        (F, b, chol_Q),
        (False, False, False)
    )

    @partial(jax.jit, backend=backend)
    def one_smoother(k, ys, kalman_smoothing_sol):
        kalman_smoothing_mean, kalman_smoothing_cov = kalman_smoothing_sol

        # Propose from filtering, weight from smoothing
        proposal_model = GaussianDensity(kalman_smoothing_mean,
                                         jax.vmap(jnp.linalg.cholesky)(kalman_smoothing_cov),
                                         )

        observation_model = LinearGaussianObservationModel(
            (H, c, chol_R, ys),
            (False, False, False, True)
        )

        initial_model = GaussianDensity(m0, chol_P0)

        if use_FFBS:
            _, sampled_indices, ell = ffbs_smoothing(k, proposal_model,
                                                     transition_model, observation_model, initial_model,
                                                     NullPotentialModel(), N=N)
            return ell, sampled_indices
        else:
            ps_result = smoothing(k, proposal_model, proposal_model,
                                  transition_model, observation_model, initial_model,
                                  NullPotentialModel(), systematic, N=N)

            return ps_result.ells[-1], ps_result.origins

    runtimes = np.empty((len(JAX_KEYS),))
    _, ps_origins = one_smoother(JAX_KEYS[0], ys, (kalman_smoothing_means, kalman_smoothing_covs))
    for i, jax_key in enumerate(JAX_KEYS):
        tic = time.time()
        _, ps_origins = one_smoother(jax_key, ys, (kalman_smoothing_means, kalman_smoothing_covs))
        ps_origins.block_until_ready()
        toc = time.time()
        runtimes[i] = toc - tic

    return np.mean(runtimes)


shape = (len(Ts), len(Ns))
runtime_means = np.empty(shape)

indices = np.recarray(shape,
                      dtype=[("T", int), ("N", int)])
for (m, T), (n, N) in product(
        *map(enumerate, [Ts, Ns]), total=reduce(mul, shape)):
    runtime = experiment(T, N)

    runtime_means[m, n] = runtime
    indices[m, n]["T"] = T
    indices[m, n]["N"] = N

os.makedirs("./output", exist_ok=True)
np.savez(f"./output/result_runtime-{use_FFBS}-{backend}", indices=indices, runtime_means=runtime_means)
