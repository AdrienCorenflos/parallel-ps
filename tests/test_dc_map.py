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

from collections import namedtuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit
from jax.config import config

from parallel_ps.core import dc_vmap, dc_pmap


@pytest.fixture(scope="session", autouse=True)
def pytest_config():
    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("np_seed", [42, 123, 666])
@pytest.mark.parametrize("pmap", [True, False])
def test_dc_map(np_seed, pmap):
    some_structure = namedtuple("some_structure", ["x", "y"])
    np.random.seed(np_seed)

    @jit
    def add(a, b):
        a_x, a_y = a
        b_x, b_y = b
        b_x = b_x + a_x[-1]
        a_y = a_y + b_y[0]

        x = jnp.concatenate([a_x, b_x], 0)
        y = jnp.concatenate([a_y, b_y], 0)
        return some_structure(x, y)

    linspace = np.arange(2, 10)

    for i, k in enumerate(linspace):
        T = 2 ** k
        x_init = np.random.randn(T, 3)
        y_init = np.random.randn(T, 4)
        elems = some_structure(x_init, y_init)
        if not pmap:
            result = dc_vmap(elems, add)
        else:
            with chex.fake_pmap():
                result = dc_pmap(elems, add, jax.devices("cpu"))

        np.testing.assert_allclose(result.x, np.cumsum(x_init, axis=0), atol=1e-5)
        np.testing.assert_allclose(result.y[::-1], np.cumsum(y_init[::-1], axis=0), atol=1e-5)
