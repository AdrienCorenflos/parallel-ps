"""
This file is adapted from my contribution to https://github.com/blackjax-devs/blackjax,
frozen from commit 08ea201796b93a79f13ac3a175731b0c858a27a4.

The changelog is as follows:
1. Use check for PRNGKey
2. Allow for different inner resamplers in residual.

This will completely be removed in favour of a simple import when BlackJAX is released as a package.

Copyright 2020 The Blackjax developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey

RESAMPLING_SIGNATURE = Callable[[jnp.ndarray, chex.PRNGKey, int], jnp.ndarray]


def _resampling_func(func, name, desc="", additional_params="") -> Callable:
    # Decorator for resampling function

    doc = f""" {name} resampling. {desc}
    Parameters
    ----------
    weights: jnp.ndarray
        Weights to resample
    key: jnp.ndarray
        PRNGKey to use in resampling
    n_samples: int
        Total number of particles to sample
    {additional_params}
    Returns
    -------
    idx: jnp.ndarray
        Array of integers fo size `n_samples` to use for resampling
    """

    func.__doc__ = doc
    return func


@partial(_resampling_func, name="Systematic")
def systematic(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int) -> jnp.ndarray:
    return _systematic_or_stratified(weights, rng_key, n_samples, True)


@partial(_resampling_func, name="Stratified")
def stratified(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int) -> jnp.ndarray:
    return _systematic_or_stratified(weights, rng_key, n_samples, False)


@partial(
    _resampling_func,
    name="Multinomial",
    desc="This has higher variance than other resampling schemes, "
         "and should only be used for illustration purposes, "
         "or if your algorithm *REALLY* needs independent samples.",
)
@partial(jax.jit, static_argnums=(2,), donate_argnums=(0, 1))
def multinomial(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int) -> jnp.ndarray:
    # In practice we don't have to sort the generated uniforms, but searchsorted works faster and is more stable
    # if both inputs are sorted, so we use the _sorted_uniforms from N. Chopin, but still use searchsorted instead of
    # his O(N) loop as our code is meant to work on GPU where searchsorted is O(log(N)) anyway.

    n = weights.shape[0]
    linspace = jax.random.uniform(rng_key, (n_samples,))
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


@partial(_resampling_func, name="Residual")
@partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0, 1))
def residual(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int,
             inner_resampling: Callable = multinomial) -> jnp.ndarray:
    # This code is adapted from nchopin/particles library, but made to be compatible with JAX static shape jitting that
    # would not have supported the dynamic slicing implementation of Nicolas. The below will be (slightly) less
    # efficient on CPU but has the benefit of being all XLA-devices compatible. The main difference with Nicolas's code
    # lies in the introduction of N+1 in the array as a "sink state" for unused indices. Sadly this can't reuse the code
    # for low variance resampling methods as it is not compatible with the sorted approach taken.

    N = weights.shape[0]

    n_samples_weights = n_samples * weights
    idx = jnp.arange(n_samples)

    integer_part = jnp.floor(n_samples_weights).astype(jnp.int32)
    sum_integer_part = jnp.sum(integer_part)

    residual_part = n_samples_weights - integer_part

    resampling_key, permutation_key = jax.random.split(rng_key)
    residual_sample = inner_resampling(residual_part / (n_samples - sum_integer_part), resampling_key, n_samples)
    residual_sample = jax.random.permutation(permutation_key, residual_sample)

    integer_idx = jnp.repeat(
        jnp.arange(N + 1),
        jnp.concatenate([integer_part, jnp.array([n_samples - sum_integer_part])], 0),
        total_repeat_length=n_samples,
    )

    idx = jnp.where(idx >= sum_integer_part, residual_sample, integer_idx)

    return idx


def coupled_resampling(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int,
                       inner_resampling: Callable = multinomial) -> jnp.ndarray:
    """
    Coupled resampling scheme that works for an arbitrary number of weights lists.

    Parameters
    ----------
    weights: jnp.ndarray
        The weights that result in coupled resampling, the first dimension is the batching dimension,
        second is the sample size
    rng_key: chex.PRNGKey
        The JAX random key for the operation
    n_samples: int
        Number of samples required
    inner_resampling: callable
        The resampling used for common part and the residual part

    Returns
    -------
    ancestors: jnp.ndarray
        The list of coupled ancestors. The first dimension is the batching dimension,
        second is the sample size

    References
    ----------
    [1] P. E. Jacob, F. Lindsten, T. B. Schön. Coupling of Particle Filters. https://arxiv.org/abs/1606.01156
    """

    batch_size, total_samples = weights.shape
    uniforms_key, coupled_key, not_coupled_key = jax.random.split(rng_key, 3)

    nu = jnp.min(weights, 0)
    alpha = jnp.sum(nu)
    mu = nu / alpha

    residuals = (weights - nu[None, :]) / (1 - alpha)

    # sample from the coupling mixture
    uniforms = jax.random.uniform(uniforms_key, (n_samples,))
    coupled = uniforms < alpha

    # where the two are coupled, simply sample from mu
    where_coupled = inner_resampling(mu, coupled_key, n_samples)

    # otherwise sample from the residuals with a common random number
    not_coupled_keys_batch = jax.random.split(not_coupled_key, batch_size)
    where_not_coupled = jax.vmap(inner_resampling, in_axes=[0, 0, None])(residuals, not_coupled_keys_batch, n_samples)

    return jnp.where(coupled[None, :], where_coupled, where_not_coupled)


@partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0, 1))
def _systematic_or_stratified(
        weights: jnp.ndarray, rng_key: PRNGKey, n_sampled: int, is_systematic: bool
) -> jnp.ndarray:
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(rng_key, ())
    else:
        u = jax.random.uniform(rng_key, (n_sampled,))
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(n_sampled, dtype=weights.dtype) + u) / n_sampled
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)

