#  MIT License
#
#  Copyright (c) 2021 Adrien Corenflos
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parallel_ps.base import DSMCState
from parallel_ps.core.resampling import multinomial, systematic, stratified, coupled_resampling


@pytest.fixture(scope="session", autouse=True)
def pytest_config():
    jax.config.update("jax_platform_name", "cpu")


LIST_RESAMPLINGS = [multinomial, systematic, stratified]


def _make_inputs(key, T, N, d):
    trajectories = np.random.randn(T, N, d)
    log_weights = np.random.rand(T, N)
    ells = np.random.randn(T, N)
    origins = np.random.randint(0, N, (T, N))

    keys = jax.random.split(key, T)
    parameters = None

    return DSMCState(trajectories, log_weights, ells, origins), keys, parameters


@pytest.mark.parametrize("B", [2, 5])
@pytest.mark.parametrize("N", [50, 100])
@pytest.mark.parametrize("np_seed", [0, 42])
@pytest.mark.parametrize("jax_seed", [1234])
@pytest.mark.parametrize("resampling", LIST_RESAMPLINGS)
def test_coupled_resampling(B, N, np_seed, jax_seed, resampling):
    stat_batch_size = 100_000
    n_samples = 2 * N // 3
    np.random.seed(np_seed)
    jax_key = jax.random.PRNGKey(jax_seed)
    weights = np.random.rand(B, N)
    weights /= weights.sum(1, keepdims=True)
    batch_keys = jax.random.split(jax_key, stat_batch_size)

    coupled_ancestors = jax.vmap(coupled_resampling, in_axes=[None, 0, None])(weights, batch_keys, n_samples)

    count = jax.vmap(lambda z: jnp.bincount(jnp.ravel(z), length=N))(jnp.swapaxes(coupled_ancestors, 0, 1))
    actual_proba = count / (stat_batch_size * n_samples)
    np.testing.assert_allclose(np.sum(actual_proba, 1), 1., atol=1e-5)
    np.testing.assert_allclose(actual_proba, weights, atol=1e-4, rtol=1e-2)
