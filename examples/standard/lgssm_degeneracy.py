"""
Experiment for degeneracy. We make the signal to noise ratio change and measure log likelihoods.
"""
import os
from functools import reduce, partial
from operator import mul

import jax
import jax.numpy as jnp
import numpy as np
from parsmooth.sequential import ekf, eks  # sequential cause we need the log-likelihood just for the experiment.
from parsmooth.utils import MVNormalParameters
from tqdm.contrib.itertools import product

from examples.models.lgssm import LinearGaussianObservationModel, LinearGaussianTransitionModel, get_data_jax
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
rs = np.logspace(-2, 2, 100)

use_FFBS = True

# data seed
data_key = jax.random.PRNGKey(0)
JAX_KEYS = jax.random.split(jax.random.PRNGKey(42), n_smoothers)


def make_model(r):
    sigma_y = 1.
    sigma_x = r * sigma_y
    F = 0.5 * np.eye(1)
    H = np.eye(dim_y, dim_x)
    b = np.zeros((dim_x,))
    c = np.zeros((dim_y,))

    chol_Q = sigma_x * np.eye(dim_x)
    chol_R = sigma_y * np.eye(dim_y)

    m0 = np.zeros((dim_x,))
    chol_P0 = sigma_x * np.eye(dim_x)
    return F, H, b, c, chol_Q, chol_R, m0, chol_P0


@partial(jax.jit, backend=backend, static_argnums=(0, 1))
def experiment(T, N):
    def r_experiment(r):
        F, H, b, c, chol_Q, chol_R, m0, chol_P0 = make_model(r)
        Q = chol_Q @ chol_Q.T
        R = chol_R @ chol_R.T
        P0 = chol_P0 @ chol_P0.T

        xs, ys = get_data_jax(data_key, m0, chol_P0, F, H, chol_R, chol_Q, b, c, T)

        transition_function = lambda x: F @ x + b
        observation_function = lambda x: H @ x + c

        x0 = MVNormalParameters(m0, P0)
        jitted_ekf = jax.jit(ekf, static_argnums=(2, 4, 6, 7))
        jitted_eks = jax.jit(eks, static_argnums=(0, 3, 4))

        kalman_ell, kalman_filtering_solution = jitted_ekf(x0, ys,
                                                           transition_function,
                                                           Q,
                                                           observation_function,
                                                           R,
                                                           None, True)

        kalman_smoothing_solution = jitted_eks(transition_function, Q,
                                               kalman_filtering_solution,
                                               None, True)

        kalman_smoothing_mean, kalman_smoothing_cov = kalman_smoothing_solution

        transition_model = LinearGaussianTransitionModel(
            (F, b, chol_Q),
            (False, False, False)
        )

        def one_experiment(k, ):

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
                return ell
            else:
                ps_result = smoothing(k, proposal_model, proposal_model,
                                      transition_model, observation_model, initial_model,
                                      NullPotentialModel(), systematic, N=N)

                return ps_result.ells[-1]

        def experiment_body(_, exp_key):
            res = one_experiment(exp_key)
            return None, res

        _, experiments_res = jax.lax.scan(experiment_body, None, JAX_KEYS)
        return experiments_res, kalman_ell

    def r_experiment_body(_, r):
        return None, r_experiment(r)

    _, res = jax.lax.scan(r_experiment_body, None, rs)
    return res


shape = (len(Ts), len(Ns))
all_ells = np.empty(shape + (len(rs), n_smoothers))
all_kf_ells = np.empty(shape + (len(rs),))

indices = np.recarray(shape,
                      dtype=[("T", int), ("N", int)])
for (m, T), (n, N) in product(
        *map(enumerate, [Ts, Ns]), total=reduce(mul, shape)):
    all_ells[m, n], all_kf_ells[m, n] = experiment(T, N)
    indices[m, n]["T"] = T
    indices[m, n]["N"] = N

os.makedirs("./output", exist_ok=True)
np.savez(f"./output/result_degeneracy-{use_FFBS}-{backend}", rs=rs, indices=indices, ells=all_ells, kf_ells=all_kf_ells)
