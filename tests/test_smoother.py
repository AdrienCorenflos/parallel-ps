import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from parallel_ps.base import DensityModel, PyTree, NullPotentialModel, DSMCState
from parallel_ps.core.resampling import systematic
from parallel_ps.smoother import smoothing as particle_smoothing
from parsmooth import FunctionalModel, MVNSqrt, filtering, smoothing, sampling
from parsmooth.linearization import extended
from .lgssm import LinearGaussianObservationModel, LinearGaussianTransitionModel, get_data, mvn_logprob_fn


@pytest.fixture(scope="session", autouse=True)
def pytest_config():
    jax.config.update("jax_disable_jit", False)


class IndependentPropsosalModel(DensityModel):
    @classmethod
    def log_potential(cls, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mean, chol = parameter
        return mvn_logprob_fn(particles, mean, chol)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        means, chols = self.parameters
        normals = jax.random.normal(key, (self.T, N, means.shape[-1]))
        return means[:, None, :] + jnp.einsum("...ij,...kj->...ki", chols, normals)


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_y", [1, 2])
@pytest.mark.parametrize("T", [100])
@pytest.mark.parametrize("np_seed", [42, 1234])
@pytest.mark.parametrize("jax_seed", [0, 31415])
@pytest.mark.parametrize("N", [50])
@pytest.mark.parametrize("conditional", [True, False])
def test_smoother(dim_x, dim_y, T, np_seed, N, jax_seed, conditional):
    np.random.seed(np_seed)
    rng_key = jax.random.PRNGKey(jax_seed)

    F = 0.95 * np.eye(dim_x)
    H = np.random.randn(dim_y, dim_x)

    b = np.random.randn(dim_x)
    c = np.random.randn(dim_y)

    chol_Q = 1e-2 * np.eye(dim_x)
    chol_R = 1e-1 * np.eye(dim_y)

    m0 = -np.linalg.solve(F - np.eye(dim_x), b)  # stationary mean
    chol_P0 = 1e-2 * np.eye(dim_x)

    xs, ys = get_data(m0, chol_P0, F, H, chol_R, chol_Q, b, c, T)

    kalman_transition_model = FunctionalModel(lambda x, eps: F @ x + eps, MVNSqrt(b, chol_Q))
    kalman_observation_model = FunctionalModel(lambda x, eps: H @ x + eps, MVNSqrt(c, chol_R))
    kalman_filtering_solution = filtering(ys, MVNSqrt(m0, chol_P0), kalman_transition_model,
                                          kalman_observation_model, extended)

    kalman_smoothing_solution = smoothing(kalman_transition_model, kalman_filtering_solution, extended)

    observation_model = LinearGaussianObservationModel(
        (H, c, chol_R, ys),
        (False, False, False, True)
    )
    transition_model = LinearGaussianTransitionModel(
        (F, b, chol_Q),
        (False, False, False)
    )

    independent_proposal_model = IndependentPropsosalModel(kalman_filtering_solution, MVNSqrt(True, True), T + 1)
    weight_model = IndependentPropsosalModel(kalman_filtering_solution, MVNSqrt(True, True), T + 1)

    initial_model = IndependentPropsosalModel(MVNSqrt(m0, chol_P0),
                                              MVNSqrt(False, False), None)

    if conditional:
        rng_key, gibbs_key = jax.random.split(rng_key)
        init_trajectory = sampling(rng_key, 1, kalman_transition_model, kalman_filtering_solution,
                                   extended)[:, 0]

        def gibbs(n_iter):
            keys = jax.random.split(rng_key, n_iter)

            def body(traj, op_key):
                sample_key, randint_key = jax.random.split(op_key)
                final_state = particle_smoothing(sample_key, independent_proposal_model, weight_model,
                                                 transition_model, observation_model, initial_model,
                                                 NullPotentialModel(), N=N, conditional_trajectory=traj)
                new_traj = final_state.trajectories[:, jax.random.randint(randint_key, (), 0, N - 1)]
                return new_traj, new_traj

            _, res = jax.lax.scan(body, init_trajectory, keys)
            return jnp.swapaxes(res, 0, 1)

        smoother_solution = gibbs(500)

    else:
        smoother_solution = particle_smoothing(rng_key, independent_proposal_model, weight_model,
                                               transition_model, observation_model, initial_model,
                                               NullPotentialModel(), systematic, N=N).trajectories

    plt.plot(smoother_solution[..., 0].mean(1), label="PS-indep", color="C0")
    plt.fill_between(np.arange(0, T+1),
                     smoother_solution[..., 0].mean(1) - 2 * smoother_solution[..., 0].std(1),
                     smoother_solution[..., 0].mean(1) + 2 * smoother_solution[..., 0].std(1), alpha=0.33, color="C0")
    plt.plot(kalman_smoothing_solution[0][:, 0], label="KS", color="C1")
    plt.fill_between(np.arange(0, T+1),
                     smoother_solution[..., 0].mean(1) - 2 * np.abs(kalman_smoothing_solution.chol[..., 0, 0]),
                     smoother_solution[..., 0].mean(1) + 2 * np.abs(kalman_smoothing_solution.chol[..., 0, 0]),
                     alpha=0.33, color="C1")
    plt.plot(kalman_filtering_solution[0][:, 0], label="KF", color="C2")
    plt.legend()
    plt.show()
