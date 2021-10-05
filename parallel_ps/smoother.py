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

from functools import partial
from typing import Callable

import chex
import jax.numpy as jnp
from jax import jit, tree_flatten, tree_unflatten, vmap
from jax.random import split


def _common():
    pass



def _initialize(key: chex.PRNGKey, q_t):


    pass


def generic_smoother(key, qt: Callable, log_qt: Callable, log_nut: Callable, log_Mt: Callable, log_Gt: Callable,
                     log_G0: Callable, log_M0: Callable, T, operator, qt_params, nut_params, Mt_params,
                     Gt_params, N=100, return_intermediary=False, operator_has_auxiliary_outputs=False,
                     passthrough_operator=None):
    key, init_key = split(key, 2)
    init_keys = split(init_key, T)
    qt = vmap(qt, in_axes=[0, None, 0])

    init_particles = qt(init_keys, N, qt_params)
    init_ells = jnp.zeros((T,))

    init_log_qt = vmap(log_qt, in_axes=[0, 0])(init_particles, qt_params)  # vmapped across time

    logw0 = vmap(log_G0)(init_particles[0]) + vmap(log_M0)(init_particles[0]) - init_log_qt[0]
    ell_0, _ = normalize(logw0)

    flat_nut_params, nut_tree_def = tree_flatten(nut_params)
    init_log_nut = vmap(log_nut, in_axes=[0, 0])(init_particles[1:],
                                                 tree_unflatten(nut_tree_def, [par[1:] for par in
                                                                               flat_nut_params]))  # vmapped across time
    log_weights = jnp.concatenate([jnp.expand_dims(logw0, 0), init_log_nut - init_log_qt[1:]])
    # init_ells = jnp.concatenate([jnp.reshape(ell_0, (1,)), init_ells[1:]])
    if T == 1:
        return init_ells, init_particles

    combination_keys = split(key, T)

    elems = combination_keys, init_particles, log_weights, init_ells, (nut_params, Mt_params, Gt_params)

    def logwt_fn(x_t, x_t_1, params_t):
        log_Gt_weight = log_Gt(x_t, x_t_1, params_t[2])
        log_Mt_weight = log_Mt(x_t, x_t_1, params_t[1])
        log_nut_weight = log_nut(x_t, params_t[0])
        return log_Gt_weight + log_Mt_weight - log_nut_weight

    logwt_fn = jnp.vectorize(logwt_fn, excluded=(2,), signature="(d),(d)->()")

    operator = jit(partial(operator, logwt_fn=logwt_fn))

    if operator_has_auxiliary_outputs:
        (_, particles, _, ells, *_), aux = compile_efficient_combination(elems, operator, return_intermediary,
                                                                         operator_has_auxiliary_outputs,
                                                                         passthrough_operator)
        ells = jnp.concatenate([jnp.reshape(ell_0, (1,)), ells[1:]])
        return ells, particles, aux
    else:
        _, particles, _, ells, *_ = compile_efficient_combination(elems, operator, return_intermediary,
                                                                  operator_has_auxiliary_outputs,
                                                                  passthrough_operator)
        ells = jnp.concatenate([jnp.reshape(ell_0, (1,)), ells[1:]])

        return ells, particles


def conditional_smoother(key, qt: Callable, log_qt: Callable, log_nut: Callable, log_Mt: Callable, log_Gt: Callable,
                         log_G0: Callable, log_M0: Callable, T, qt_params, nut_params, Mt_params,
                         Gt_params, x_t_cond, N=100):
    key, init_key = split(key, 2)
    init_keys = split(init_key, T)
    qt = vmap(qt, in_axes=[0, None, 0])

    init_particles = qt(init_keys, N - 1, qt_params)
    init_particles = jnp.concatenate([jnp.expand_dims(x_t_cond, 1), init_particles], axis=1)
    init_ells = jnp.zeros((T,))

    init_log_qt = vmap(log_qt, in_axes=[0, 0])(init_particles, qt_params)  # vmapped across time

    logw0 = vmap(log_G0)(init_particles[0]) + vmap(log_M0)(init_particles[0]) - init_log_qt[0]
    ell_0, _ = normalize(logw0)

    flat_nut_params, nut_tree_def = tree_flatten(nut_params)
    init_log_nut = vmap(log_nut, in_axes=[0, 0])(init_particles[1:],
                                                 tree_unflatten(nut_tree_def, [par[1:] for par in
                                                                               flat_nut_params]))  # vmapped across time
    log_weights = jnp.concatenate([jnp.expand_dims(logw0, 0), init_log_nut - init_log_qt[1:]])
    # init_ells = jnp.concatenate([jnp.reshape(ell_0, (1,)), init_ells[1:]])
    if T == 1:
        return init_ells, init_particles

    combination_keys = split(key, T)

    elems = combination_keys, init_particles, log_weights, init_ells, (nut_params, Mt_params, Gt_params, x_t_cond)

    def logwt_fn(x_t, x_t_1, params_t):
        log_Gt_weight = log_Gt(x_t, x_t_1, params_t[2])
        log_Mt_weight = log_Mt(x_t, x_t_1, params_t[1])
        log_nut_weight = log_nut(x_t, params_t[0])
        return log_Gt_weight + log_Mt_weight - log_nut_weight

    logwt_fn = jnp.vectorize(logwt_fn, excluded=(2,), signature="(d),(d)->()")

    operator = jit(partial(conditional_n2_combination, logwt_fn=logwt_fn, resampling_method=multinomial))

    _, particles, _, ells, *_ = compile_efficient_combination(elems, operator)

    return ells, particles
