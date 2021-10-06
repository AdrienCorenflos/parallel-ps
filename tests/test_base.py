import chex
import numpy as np
import pytest
from scipy.special import logsumexp

from parallel_ps.base import rejoin_batched_and_static_params, split_batched_and_static_params, normalize


def test_split_rejoin():
    batched = dict(x=True, y=False)
    params = dict(x=np.random.random(10), y=np.random.random(5))
    batched_params, static_params = split_batched_and_static_params(params, batched)
    rejoined_params = rejoin_batched_and_static_params(batched, batched_params, static_params)
    chex.assert_tree_all_close(params, rejoined_params)


def test_normalize():
    np.random.seed(42)
    log_weights = -np.random.rand(54321)

    weights = normalize(log_weights)
    assert np.sum(weights) == pytest.approx(1, 1e-6, 1e-6)
    assert np.mean(log_weights - np.log(weights)) == pytest.approx(logsumexp(log_weights), 1e-6, 1e-6)
