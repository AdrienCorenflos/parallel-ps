import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

jax.config.update("jax_disable_jit", False)

from parallel_ps.base import DensityModel, PyTree, UnivariatePotentialModel
from parallel_ps.core.resampling import systematic
from parallel_ps.doc_util import doc_inherit
from parallel_ps.smoother import smoothing as particle_smoothing
from parsmooth import FunctionalModel, MVNSqrt, sampling, filtering, smoothing
from parsmooth.linearization import extended
from .lgssm import LinearGaussianObservationModel, LinearGaussianTransitionModel, get_data, mvn_logprob_fn


@pytest.fixture(scope="session", autouse=True)
def pytest_config():
    jax.config.update("jax_disable_jit", False)


class IndependentPropsosalModel(DensityModel):
    parameters: PyTree
    batched: PyTree
    T: int

    @classmethod
    def log_potential(cls, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mean, chol = parameter
        return mvn_logprob_fn(particles, mean, chol)

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        means, chols = self.parameters
        normals = jax.random.normal(key, (self.T, N, means.shape[-1]))
        return means[:, None, :] + jnp.einsum("...ij,...kj->...ki", chols, normals)


class PathwiseSamplerModel(DensityModel):
    parameters: PyTree
    batched: PyTree
    T: int

    transition_model: FunctionalModel
    filter_trajectory: FunctionalModel

    @classmethod
    @doc_inherit
    def log_potential(cls, particles: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        mean, chol = parameter
        return mvn_logprob_fn(particles, mean, chol)

    @doc_inherit
    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        return sampling(key, N, self.transition_model, self.filter_trajectory, extended, parallel=False)


class NullPotentialModel(UnivariatePotentialModel):
    @classmethod
    def log_potential(cls, particles: chex.ArrayTree, parameter: PyTree) -> float:
        return 0.


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_y", [1, 2])
@pytest.mark.parametrize("T", [60])
@pytest.mark.parametrize("np_seed", [42])
@pytest.mark.parametrize("N", [100])
def test_kalman_independent_proposals(dim_x, dim_y, T, np_seed, N):
    np.random.seed(np_seed)
    rng_key = jax.random.PRNGKey(1234)

    F = 0.9 * np.eye(dim_x)
    H = np.random.randn(dim_y, dim_x)

    b = np.random.randn(dim_x)
    c = np.random.randn(dim_y)

    chol_Q = 1e-1 * np.eye(dim_x)
    chol_R = 1e-1 * np.eye(dim_y)

    m0 = -np.linalg.solve(F-np.eye(dim_x), b)  # stationary mean
    chol_P0 = 1e-2 * np.eye(dim_x)

    xs, ys = get_data(m0, chol_P0, F, H, chol_R, chol_Q, b, c, T)

    kalman_transition_model = FunctionalModel(lambda x, eps: F @ x + eps, MVNSqrt(b, chol_Q))
    kalman_observation_model = FunctionalModel(lambda x, eps: H @ x + eps, MVNSqrt(c, chol_R))
    kalman_filtering_solution = filtering(ys, MVNSqrt(m0, chol_P0), kalman_transition_model,
                                          kalman_observation_model, extended, parallel=False)

    kalman_smoothing_solution = smoothing(kalman_transition_model, kalman_filtering_solution, extended)

    observation_model = LinearGaussianObservationModel(
        (H, c, chol_R, ys),
        (False, False, False, True)
    )
    transition_model = LinearGaussianTransitionModel(
        (F, b, chol_Q),
        (False, False, False)
    )

    proposal_model = IndependentPropsosalModel(kalman_filtering_solution, MVNSqrt(True, True), T + 1)
    weight_model = IndependentPropsosalModel(kalman_filtering_solution, MVNSqrt(True, True), T + 1)

    initial_model = IndependentPropsosalModel(MVNSqrt(m0, chol_P0),
                                              MVNSqrt(False, False), None)

    particle_smoother_solution = particle_smoothing(rng_key, proposal_model, weight_model, transition_model,
                                                    observation_model, initial_model, NullPotentialModel(None, None),
                                                    systematic, N=N)

    plt.plot(particle_smoother_solution.trajectories.mean(1)[:, 0], label="PS")
    for n in range(N):
        plt.scatter(np.arange(T+1), particle_smoother_solution.trajectories[:, n, 0], alpha=0.25)
    plt.plot(kalman_smoothing_solution[0][:, 0], label="KS")
    plt.plot(kalman_filtering_solution[0][:, 0], label="KF")
    plt.legend()
    plt.show()

    np.testing.assert_allclose(particle_smoother_solution.trajectories.mean(1),
                               kalman_smoothing_solution[0], atol=1e-2,
                               rtol=1e-2)

