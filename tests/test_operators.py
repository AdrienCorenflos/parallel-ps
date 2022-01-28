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

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parallel_ps.base import DSMCState, normalize
from parallel_ps.core.resampling import multinomial, systematic, stratified
from parallel_ps.operators import operator, lazy_operator


@pytest.fixture(scope="session", autouse=True)
def pytest_config():
    ...


LIST_RESAMPLINGS = [multinomial, systematic, stratified]


def _make_inputs(key, T, N, d):
    trajectories = 1 + np.random.randn(T, N, d)
    log_weights = np.random.rand(T, N)
    log_weights = np.log(log_weights / log_weights.sum(1, keepdims=True))
    ells = np.random.randn(T)
    origins = np.random.randint(0, N, (T, N))

    keys = jax.random.split(key, T)
    parameters = None

    return DSMCState(trajectories, log_weights, ells, origins), keys, parameters


@pytest.mark.parametrize("N", [25, 50])
@pytest.mark.parametrize("np_seed", [0, 42])
@pytest.mark.parametrize("resampling", LIST_RESAMPLINGS)
def test_operator_independent_mixing_weights(N, np_seed, resampling):
    np.random.seed(np_seed)
    key_a, key_b = jax.random.split(jax.random.PRNGKey(666))
    log_weight_function = lambda a, b, _: 1.

    inputs_a = _make_inputs(key_a, 10, N, 3)
    inputs_b = _make_inputs(key_b, 5, N, 3)

    state_out, keys_out, params_out, _ = operator(inputs_a, inputs_b, log_weight_function, resampling, N, False)
    assert params_out is None
    np.testing.assert_array_equal(keys_out, np.concatenate([inputs_a[1], inputs_b[1]]))
    chex.assert_shape(state_out.trajectories, (15, N, 3))
    chex.assert_tree_all_close(state_out.log_weights, -np.log(N))
    chex.assert_type(state_out.origins, int)

    phi = lambda z: np.sum(z ** 2, -1)

    actual = phi(state_out.trajectories).mean(-1)
    expected = 3.

    np.testing.assert_allclose(actual, expected, rtol=1e-1)


@pytest.mark.parametrize("N", [25, 50])
@pytest.mark.parametrize("np_seed", [0, 42])
@pytest.mark.parametrize("resampling", LIST_RESAMPLINGS)
def test_operator_other_mixing_weights(N, np_seed, resampling):
    np.random.seed(np_seed)
    key_a, key_b = jax.random.split(jax.random.PRNGKey(666))
    log_weight_function = lambda a, b, _: -jnp.sum((a - b) ** 4, -1)

    state_a, _, params_a = _make_inputs(key_a, 4, N, 3)
    state_b, _, params_b = _make_inputs(key_b, 5, N, 3)

    phi = lambda z: np.sum(z ** 4, -1)

    @jax.vmap
    def vmapped_operator(k):
        keys = jax.random.split(k, 9)
        keys_a = keys[:4]
        keys_b = keys[4:]

        inputs_a = state_a, keys_a, params_a
        inputs_b = state_b, keys_b, params_b

        state_out, keys_out, params_out, _ = operator(inputs_a, inputs_b, log_weight_function, resampling, N, False)
        return phi(state_out.trajectories).mean(-1)

    batch_key = jax.random.split(jax.random.PRNGKey(666), 100)

    actual = vmapped_operator(batch_key)
    actual_mean = actual.mean(0)

    # compute expected as an importance sample
    log_weights = log_weight_function(state_a.trajectories[-1][None, ...],
                                      state_b.trajectories[0][:, None, :],
                                      None)

    log_weights = log_weights + state_a.log_weights[-1, :, None] + state_b.log_weights[0, None, :]

    weights = np.asarray(np.reshape(normalize(np.ravel(log_weights)), (N, N)))
    w_a, w_b = weights.sum(0), weights.sum(1)
    expected_a = np.average(phi(state_a.trajectories), axis=-1, weights=w_a)
    expected_b = np.average(phi(state_b.trajectories), axis=-1, weights=w_b)

    np.testing.assert_allclose(actual_mean[:4], expected_a, atol=1e-2, rtol=1e-1)
    np.testing.assert_allclose(actual_mean[-5:], expected_b, atol=1e-2, rtol=1e-1)


@pytest.mark.parametrize("N", [1_000])
@pytest.mark.parametrize("np_seed", [0, 42])
def test_lazy_operator(N, np_seed):
    np.random.seed(np_seed)
    key_a, key_b = jax.random.split(jax.random.PRNGKey(666))
    log_weight_function = lambda a, b, _: -jnp.sum((a - b) ** 4, -1)

    state_a, _, params_a = _make_inputs(key_a, 4, N, 3)
    state_b, _, params_b = _make_inputs(key_b, 5, N, 3)

    phi = lambda z: np.sum(z ** 4, -1)

    @jax.vmap
    def vmapped_operator(k):
        keys = jax.random.split(k, 9)
        keys_a = keys[:4]
        keys_b = keys[4:]

        log_weights_a = state_a.log_weights[-1]
        log_weights_b = state_b.log_weights[0]

        max_log_weights_a = jnp.max(log_weights_a)
        max_log_weights_b = jnp.max(log_weights_b)

        inputs_a = state_a, keys_a, params_a, jnp.nan * jnp.ones((4,))
        inputs_b = state_b, keys_b, params_b, (max_log_weights_a + max_log_weights_b) * jnp.ones((4,))

        state_out, keys_out, params_out, _ = lazy_operator(inputs_a, inputs_b, log_weight_function, N)
        return phi(state_out.trajectories).mean(-1), state_out

    batch_key = jax.random.split(jax.random.PRNGKey(666), 100)

    actual, states_out = vmapped_operator(batch_key)

    actual_mean = actual.mean(0)

    # compute expected as an importance sample
    log_weights = log_weight_function(state_a.trajectories[-1][None, ...],
                                      state_b.trajectories[0][:, None, :],
                                      None)

    log_weights = log_weights + state_a.log_weights[-1, :, None] + state_b.log_weights[0, None, :]

    weights = np.asarray(np.reshape(normalize(np.ravel(log_weights)), (N, N)))
    w_a, w_b = weights.sum(0), weights.sum(1)
    expected_a = np.average(phi(state_a.trajectories), axis=-1, weights=w_a)
    expected_b = np.average(phi(state_b.trajectories), axis=-1, weights=w_b)

    np.testing.assert_allclose(actual_mean[:4], expected_a, atol=1, rtol=1e-1)
    np.testing.assert_allclose(actual_mean[-5:], expected_b, atol=1, rtol=1e-1)
