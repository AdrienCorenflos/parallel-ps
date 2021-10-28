# from tqdm.contrib.itertools import product
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from parsmooth.sequential import ekf, eks  # sequential cause we need the log-likelihood just for the experiment.
from parsmooth.utils import MVNormalParameters
from scipy.stats import describe

from examples.models.lgssm import get_data, LinearGaussianObservationModel, LinearGaussianTransitionModel
from parallel_ps.base import GaussianDensity, NullPotentialModel
from parallel_ps.core.resampling import systematic
from parallel_ps.parallel_smoother import smoothing

# CONFIG


n_data = 10  # number of times the SSM is sampled from per experiment
n_smoothers = 5  # number of  times we run the smoother on each dataset

# SSM Config
dims_x = [1, 2, 3]
dims_y = [1, 2, 3]

sigmas_x = np.logspace(-2, 0, 4)
sigmas_y = np.logspace(-2, 0, 4)

Ts = [100, 1_000, 10_000]
Ns = [25, 50, 100]

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
    kalman_ells, kalman_filtering_solutions = jax.vmap(ekf, in_axes=[None, 0] + [None] * 6)(x0, batch_ys,
                                                                                            transition_function, Q,
                                                                                            observation_function, R,
                                                                                            None, True)

    kalman_smoothing_solutions = jax.vmap(eks, in_axes=[None, None, 0, None, None])(transition_function, Q,
                                                                                    kalman_filtering_solutions,
                                                                                    None, True)

    kalman_smoothing_means, kalman_smoothing_covs = kalman_smoothing_solutions

    transition_model = LinearGaussianTransitionModel(
        (F, b, chol_Q),
        (False, False, False)
    )

    @jax.jit
    def one_smoother(k, ys, kalman_smoothing_sol):
        kalman_smoothing_mean, kalman_smoothing_cov = kalman_smoothing_sol

        # Propose from filtering, weight from smoothing
        proposal_model = GaussianDensity(kalman_smoothing_mean,
                                         jax.vmap(jnp.linalg.cholesky)(kalman_smoothing_cov),
                                         )

        weight_model = GaussianDensity(kalman_smoothing_mean,
                                       jax.vmap(jnp.linalg.cholesky)(kalman_smoothing_cov))

        observation_model = LinearGaussianObservationModel(
            (H, c, chol_R, ys),
            (False, False, False, True)
        )

        initial_model = GaussianDensity(m0, chol_P0)

        ps_result = smoothing(k, proposal_model, weight_model,
                              transition_model, observation_model, initial_model,
                              NullPotentialModel(), systematic, N=N)

        return ps_result.ells[-1], ps_result.origins

    ps_results = np.empty((len(JAX_KEYS), n_data))
    ps_unique_ancestors = np.empty((len(JAX_KEYS), n_data, T + 1))

    for i, jax_key in enumerate(JAX_KEYS):
        for j, (ys, smoothing_mean, smoothing_cov) in enumerate(
                zip(batch_ys, kalman_smoothing_means, kalman_smoothing_covs)):
            ps_results[i, j], ps_origins = one_smoother(jax_key, ys, (smoothing_mean, smoothing_cov))
            for t in range(T+1):
                ps_unique_ancestors[i, j, t] = len(np.unique(ps_origins[t]))

    ps_ell_means = ps_results.mean(0)
    ps_ell_vars = ps_results.var(0)
    ps_unique_ancestors = ps_unique_ancestors.mean(0)

    return kalman_ells, ps_ell_means, ps_ell_vars, ps_unique_ancestors


experiments_results = dict()

for dim_x, dim_y, sigma_x, sigma_y in product(dims_x, dims_y, sigmas_x, sigmas_y):
    print(dim_x, dim_y, sigma_x, sigma_y)
    kalman_ells = np.empty((len(Ts), len(Ns), n_data))
    ps_ell_means = np.empty((len(Ts), len(Ns), n_data))
    ps_ell_vars = np.empty((len(Ts), len(Ns), n_data))
    experiments_results[(dim_x, dim_y, sigma_x, sigma_y)] = kalman_ells, ps_ell_means, ps_ell_vars
    # for (i, T), (j, N) in product(enumerate(Ts), enumerate(Ns), leave=False, total=len(Ts) * len(Ns)):  # tqdm version
    for (i, T), (j, N) in product(enumerate(Ts), enumerate(Ns)):
        result = experiment(dim_x, dim_y, sigma_x, sigma_y, T, N)
        kalman_ells[i, j, :], ps_ell_means[i, j, :], ps_ell_vars[i, j, :], ps_unique_ancestors = result
        print(T, N)
        print("Relative diff mean", np.mean((kalman_ells[i, j, :] - ps_ell_means[i, j, :]) / kalman_ells[i, j, :],
                                            axis=-1))
        print("Diff mean", np.mean((kalman_ells[i, j, :] - ps_ell_means[i, j, :]), axis=-1))
        print("PS ell std", np.mean(ps_ell_vars[i, j, :] ** 0.5, axis=-1))
        print("Diff mean corrected", np.mean((kalman_ells[i, j, :] - ps_ell_means[i, j, :] - ps_ell_vars[i, j, :] / 2),
                                             axis=-1))
        print("Unique origins stats", ps_unique_ancestors.mean(), ps_unique_ancestors.min(), ps_unique_ancestors.max())

    print()


# for dim_x, dim_y, sigma_x, sigma_y in product(dims_x, dims_y, sigmas_x, sigmas_y):
#     result = experiments_results[(dim_x, dim_y, sigma_x, sigma_y)]
#     kalman_ells, ps_ell_means, ps_ell_vars = result
#
#     print(dim_x, dim_y, sigma_x, sigma_y)
#     print(np.mean((kalman_ells - ps_ell_means) / kalman_ells, axis=-1))
#     print(np.mean((kalman_ells - ps_ell_means), axis=-1))
#     print(np.std(ps_ell_means, axis=-1))
#     print(np.mean((kalman_ells - ps_ell_means + ps_ell_vars / 2), axis=-1))
#     print()
