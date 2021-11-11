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

# from itertools import product

# CONFIG

backend = "gpu"
n_data = 10  # number of times the SSM is sampled from per experiment
n_smoothers = 5  # number of  times we run the smoother on each dataset

# SSM Config
dims_x = [1, 2, 3]
dims_y = [1, 2, 3]

sigmas_x = np.logspace(-2, 0, 4)
sigmas_y = np.logspace(-2, 0, 4)

Ts = [2 ** k - 1 for k in range(2, 14, 3)]
Ns = [25, 50]
use_FFBS = False

# data seed
np.random.seed(0)
JAX_KEYS = jax.random.split(jax.random.PRNGKey(42), n_smoothers)


def make_model(dim_x, dim_y, sigma_x, sigma_y):
    F = 0.9 * np.eye(dim_x)
    H = np.eye(dim_y, dim_x)
    b = np.zeros((dim_x,))
    c = np.zeros((dim_y,))

    chol_Q = sigma_x * np.eye(dim_x)
    chol_R = sigma_y * np.eye(dim_y)

    m0 = np.zeros((dim_x,))
    chol_P0 = sigma_x * np.eye(dim_x)
    return F, H, b, c, chol_Q, chol_R, m0, chol_P0


def experiment(dim_x, dim_y, sigma_x, sigma_y, T, N):
    F, H, b, c, chol_Q, chol_R, m0, chol_P0 = make_model(dim_x, dim_y, sigma_x, sigma_y)

    Q = chol_Q @ chol_Q.T
    R = chol_R @ chol_R.T
    P0 = chol_P0 @ chol_P0.T

    batch_xs = np.empty((n_data, T + 1, dim_x))
    batch_ys = np.empty((n_data, T, dim_y))

    for k in range(n_data):
        batch_xs[k], batch_ys[k] = get_data(m0, chol_P0, F, H, chol_R, chol_Q, b, c, T)

    transition_function = lambda x: F @ x + b
    observation_function = lambda x: H @ x + c

    x0 = MVNormalParameters(m0, P0)
    jitted_ekf = jax.jit(ekf, static_argnums=(2, 4, 6, 7), backend="cpu")
    jitted_eks = jax.jit(eks, static_argnums=(0, 3, 4), backend="cpu")

    kalman_ells, kalman_filtering_solutions = jax.vmap(jitted_ekf, in_axes=[None, 0] + [None] * 6)(x0, batch_ys,
                                                                                                   transition_function,
                                                                                                   Q,
                                                                                                   observation_function,
                                                                                                   R,
                                                                                                   None, True)

    kalman_smoothing_solutions = jax.vmap(jitted_eks, in_axes=[None, None, 0, None, None])(transition_function, Q,
                                                                                           kalman_filtering_solutions,
                                                                                           None, True)

    kalman_smoothing_means, kalman_smoothing_covs = kalman_smoothing_solutions

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

    ps_results = np.empty((len(JAX_KEYS), n_data))
    ps_unique_ancestors = np.empty((len(JAX_KEYS), n_data, T + 1))
    runtimes = np.empty_like(ps_results)

    for i, jax_key in enumerate(JAX_KEYS):
        for j, (ys, smoothing_mean, smoothing_cov) in enumerate(
                zip(batch_ys, kalman_smoothing_means, kalman_smoothing_covs)):
            tic = time.time()
            ps_results[i, j], ps_origins = one_smoother(jax_key, ys, (smoothing_mean, smoothing_cov))
            ps_origins.block_until_ready()
            toc = time.time()
            runtimes[i, j] = toc - tic
            for t in range(T + 1):
                ps_unique_ancestors[i, j, t] = len(np.unique(ps_origins[t]))

    ps_ell_means = ps_results.mean(0)
    ps_ell_vars = ps_results.var(0)
    ps_unique_ancestors = ps_unique_ancestors.mean(0)

    return kalman_ells, ps_ell_means, ps_ell_vars, ps_unique_ancestors, np.median(runtimes)


shape = (len(dims_x), len(dims_y), len(sigmas_x), len(sigmas_y), len(Ts), len(Ns))
kalman_ells = np.empty(shape)
ps_ell_means = np.empty(shape)
rel_ell_means = np.empty(shape)
abs_ell_means = np.empty(shape)
ps_ell_vars = np.empty(shape)
ps_ell_stds = np.empty(shape)
runtime_medians = np.empty(shape)
ps_unique_ancestors_min = np.empty(shape)
ps_unique_ancestors_mean = np.empty(shape)
ps_unique_ancestors_max = np.empty(shape)

indices = np.recarray(shape + (6,),
                      dtype=[('dim_x', int), ('dim_y', int), ("sigma_x", float), ("sigma_y", float), ("T", int),
                             ("N", int)])

for (i, dim_x), (j, dim_y), (k, sigma_x), (l, sigma_y), (m, T), (n, N) in product(
        *map(enumerate, [dims_x, dims_y, sigmas_x, sigmas_y, Ts, Ns]), total=reduce(mul, shape)):
    curr_kalman_ells, curr_ps_ell_means, curr_ps_ell_vars, ps_unique_ancestors, runtime = experiment(dim_x, dim_y,
                                                                                                     sigma_x,
                                                                                                     sigma_y, T, N)

    kalman_ells[i, j, k, l, m, n] = np.mean(curr_kalman_ells)
    ps_ell_means[i, j, k, l, m, n] = np.mean(curr_ps_ell_means)
    rel_ell_means[i, j, k, l, m, n] = np.mean((curr_kalman_ells - curr_ps_ell_means) / curr_kalman_ells)
    abs_ell_means[i, j, k, l, m, n] = np.mean((curr_kalman_ells - curr_ps_ell_means))
    ps_ell_vars[i, j, k, l, m, n] = np.mean(curr_ps_ell_vars)
    ps_ell_stds[i, j, k, l, m, n] = np.mean(curr_ps_ell_vars ** 0.5)
    ps_unique_ancestors_min[i, j, k, l, m, n] = np.min(ps_unique_ancestors)
    ps_unique_ancestors_mean[i, j, k, l, m, n] = np.mean(ps_unique_ancestors)
    ps_unique_ancestors_max[i, j, k, l, m, n] = np.max(ps_unique_ancestors)
    runtime_medians[i, j, k, l, m, n] = runtime
    indices[i, j, k, l, m, n] = (dim_x, dim_y, sigma_x, sigma_y, T, N)

os.makedirs("./output", exist_ok=True)
np.savez(f"./output/result_degeneracy-{use_FFBS}-{backend}", indices=indices, kalman_ells=kalman_ells,
         ps_ell_means=ps_ell_means, ps_ell_vars=ps_ell_vars, ps_ell_stds=ps_ell_stds, rel_ell_means=rel_ell_means,
         ps_unique_ancestors_min=ps_unique_ancestors_min, ps_unique_ancestors_mean=ps_unique_ancestors_mean,
         ps_unique_ancestors_max=ps_unique_ancestors_max, runtime_medians=runtime_medians, abs_ell_means=abs_ell_means)
