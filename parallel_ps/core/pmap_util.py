#   MIT License
#
#  Copyright (c) Adrien Corenflos 2021
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


from functools import wraps, partial

import jax
import jax.numpy as jnp
from jax import tree_unflatten


def pmap(fun, devices):
    n_devices = len(devices)

    @wraps(fun)
    def wrapped_fun(*inputs):
        flat_inputs, tree_def = jax.tree_flatten(inputs)
        flat_shapes = jax.tree_map(jnp.shape, flat_inputs)
        shapes = tree_unflatten(tree_def, flat_shapes)
        n_elems = flat_shapes[0][0]

        if n_elems <= n_devices:
            return jax.pmap(fun, devices=devices[:n_elems])(*inputs)

        pad_size = n_devices * (n_elems // n_devices + 1) - n_elems
        reshaped_inputs = jax.tree_map(lambda z: _pad(z, pad_size), inputs)
        reshaped_inputs = jax.tree_map(lambda z, shape: jnp.reshape(z, (n_devices, -1, *shape[1:])),
                                       reshaped_inputs, shapes)
        padded_res = jax.pmap(jax.vmap(fun), devices=devices)(*reshaped_inputs)
        padded_res = jax.tree_map(lambda z: jnp.reshape(z, (-1, *z.shape[2:])), padded_res)
        return jax.tree_map(lambda z: z[: n_elems], padded_res)

    return wrapped_fun


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _pad(elem, padding_size):
    def dtype_to_value(dtype):
        if jnp.issubdtype(dtype, jnp.inexact):
            return jnp.nan
        if jnp.issubdtype(dtype, jnp.integer):
            return 0
        if jnp.issubdtype(dtype, jnp.bool_):
            return False

    return jnp.pad(elem, [(0, padding_size)] + [(0, 0)] * (len(elem.shape) - 1),
                   constant_values=dtype_to_value(elem.dtype))
