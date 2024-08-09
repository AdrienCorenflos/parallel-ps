import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from parsmooth.sequential import ekf, eks
from parsmooth.utils import MVNormalParameters

from parallel_ps.base import NullPotentialModel, GaussianDensity
from parallel_ps.core.resampling import systematic
from parallel_ps.parallel_smoother import smoothing as particle_smoothing
from parallel_ps.sequential import smoothing as sequential_smoothing
from .lgssm import LinearGaussianObservationModel, LinearGaussianTransitionModel, get_data


@pytest.fixture(scope="session", autouse=True)
def pytest_config():
    jax.config.update("jax_disable_jit", False)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_y", [1, 2])
@pytest.mark.parametrize("T", [120])
@pytest.mark.parametrize("np_seed", [1234])
@pytest.mark.parametrize("jax_seed", [42])
@pytest.mark.parametrize("N", [100])
@pytest.mark.parametrize("conditional", [True, False])
@pytest.mark.parametrize("ffbs", [True, False])
def test_smoother(dim_x, dim_y, T, np_seed, N, jax_seed, conditional, ffbs):
    np.random.seed(np_seed)
    rng_key = jax.random.PRNGKey(jax_seed)

    F = 0.9 * np.eye(dim_x)
    H = np.random.randn(dim_y, dim_x)

    b = np.random.randn(dim_x)
    c = np.random.randn(dim_y)

    chol_Q = 1e-2 * np.eye(dim_x)
    chol_R = 1e-1 * np.eye(dim_y)
    Q = chol_Q @ chol_Q.T
    R = chol_R @ chol_R.T

    m0 = -np.linalg.solve(F - np.eye(dim_x), b)  # stationary mean
    chol_P0 = 1e-1 * np.eye(dim_x)
    P0 = chol_P0 @ chol_P0.T
    x0 = MVNormalParameters(m0, P0)

    xs, ys = get_data(m0, chol_P0, F, H, chol_R, chol_Q, b, c, T)

    transition_function = lambda x: F @ x + b
    observation_function = lambda x: H @ x + c
    kalman_ell, kalman_filtering_solution = ekf(x0, ys, transition_function,
                                                Q, observation_function, R, propagate_first=True)

    kalman_smoothing_solution = eks(transition_function, Q, kalman_filtering_solution, propagate_first=True)

    observation_model = LinearGaussianObservationModel(
        (H, c, chol_R, ys),
        (False, False, False, True)
    )
    transition_model = LinearGaussianTransitionModel(
        (F, b, chol_Q),
        (False, False, False)
    )
    kalman_filtering_means, kalman_filtering_covs = kalman_filtering_solution
    kalman_smoothing_means, kalman_smoothing_covs = kalman_smoothing_solution

    # Propose from filtering, weight from smoothing
    proposal_model = GaussianDensity(kalman_filtering_means,
                                     jax.vmap(jnp.linalg.cholesky)(kalman_filtering_covs),
                                     )

    weight_model = GaussianDensity(kalman_smoothing_means,
                                   jax.vmap(jnp.linalg.cholesky)(kalman_smoothing_covs))

    initial_model = GaussianDensity(m0, chol_P0)

    if conditional:
        rng_key, gibbs_key = jax.random.split(rng_key)
        init_trajectory = proposal_model.sample(rng_key, 1)[:, 0]

        def gibbs(n_iter):
            keys = jax.random.split(rng_key, n_iter)

            def body(traj, op_key):
                sample_key, randint_key = jax.random.split(op_key)
                final_state = particle_smoothing(sample_key, proposal_model, weight_model,
                                                 transition_model, observation_model, initial_model,
                                                 NullPotentialModel(), N=N, conditional_trajectory=traj)
                new_traj = final_state.trajectories[:, jax.random.randint(randint_key, (), 0, N - 1)]
                return new_traj, new_traj

            _, res = jax.lax.scan(body, init_trajectory, keys)
            return jnp.swapaxes(res, 0, 1)

        smoother_solution = gibbs(500)
        np.testing.assert_allclose(smoother_solution[:, 100:].mean(1), kalman_smoothing_solution.mean, rtol=1e-2)
    elif ffbs:
        smoother = jax.vmap(lambda k: sequential_smoothing(T, k, transition_model, observation_model, initial_model,
                                                           NullPotentialModel(), N=N, M=N))

        (smoother_solutions, _), ells = smoother(jax.random.split(rng_key, 10))
        np.testing.assert_allclose(ells[:, -1].mean() - ells[:, -1].var() / 2,
                                   kalman_ell, rtol=1e-2)
        smoother_solution = smoother_solutions[0]
    else:

        smoother = jax.vmap(lambda k: particle_smoothing(k, proposal_model, weight_model,
                                                         transition_model, observation_model, initial_model,
                                                         NullPotentialModel(), systematic, N=N))

        smoother_solutions = smoother(jax.random.split(rng_key, 10))
        np.testing.assert_allclose(smoother_solutions.ells[:, -1].mean() - smoother_solutions.ells[:, -1].var() / 2,
                                   kalman_ell, rtol=1e-2)
        smoother_solution = smoother_solutions.trajectories[0]

    plt.plot(smoother_solution[..., 0].mean(1), label="PS-indep", color="C0")
    plt.fill_between(np.arange(0, T + 1),
                     smoother_solution[..., 0].mean(1) - 2 * smoother_solution[..., 0].std(1),
                     smoother_solution[..., 0].mean(1) + 2 * smoother_solution[..., 0].std(1), alpha=0.33, color="C0")
    plt.plot(kalman_smoothing_solution[0][:, 0], label="KS", color="C1")
    plt.fill_between(np.arange(0, T + 1),
                     smoother_solution[..., 0].mean(1) - 2 * np.abs(kalman_smoothing_solution.cov[..., 0, 0] ** 0.5),
                     smoother_solution[..., 0].mean(1) + 2 * np.abs(kalman_smoothing_solution.cov[..., 0, 0] ** 0.5),
                     alpha=0.33, color="C1")
    plt.plot(kalman_filtering_solution[0][:, 0], label="KF", color="C2")
    plt.plot(xs[:, 0], label="True states", color="C3")

    plt.legend()
    plt.suptitle(f"FFBS-{ffbs}-Gibbs-{conditional}, dim_x-{dim_x}, dim_y-{dim_y}")
    plt.show()
