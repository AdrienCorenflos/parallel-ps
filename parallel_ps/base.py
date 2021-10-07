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
from typing import TypeVar, Union, Tuple, List, Dict, NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax import numpy as jnp, tree_unflatten, tree_flatten
from jax.scipy.special import logsumexp

PyTree = TypeVar("PyTree", bound=Union[jnp.ndarray, NamedTuple, Tuple, List, Dict, bool, float, None])


def normalize(log_weights: jnp.ndarray) -> jnp.ndarray:
    """
    This utility takes unnormalised `log_weights`, and returns the normalised weights in their natural space.
    It does it in a numerically stable way similar to the logsumexp trick.

    Parameters
    ----------
    log_weights: jnp.ndarray
        The unnormalised log-weights

    Returns
    -------
    weights: jnp.array
        The resulting normalised weights
    """

    logw = log_weights - logsumexp(log_weights)
    w = jnp.exp(logw)
    return w


def ess(weights: jnp.ndarray) -> float:
    """
    Computes the relative effective sample size given normalised weights.
    This will return a bogus number if the weights are not normalised first

    Parameters
    ----------
    weights: jnp.ndarray
        Normalised weights

    Returns
    -------
    ess: float
        The effective sample size expressed as a percentage of the array size.
    """
    return 1 / jnp.square(weights).sum()


class DSMCState(NamedTuple):
    """
    Current state of the partial smoother.

    trajectories:
        Collection of partial smoothing trajectories. This can be any Pytree (see JAX doc)
    log_weights:
        This is only useful at initialisation. Otherwise, they will be constant due to the fact we resample at
        each stitching operation
    ells:
        Running log-normalizing constant of the partial smoothing distribution
    origins:
        What were the original indices of the currently alive particles for each trajectory.
    """
    trajectories: chex.ArrayTree
    log_weights: jnp.ndarray
    ells: jnp.ndarray
    origins: jnp.ndarray


@partial(jax.jit, static_argnums=(1,))
def split_batched_and_static_params(parameters, batched):
    chex.assert_trees_all_equal_structs(parameters, batched)

    flat_params, tree_def = tree_flatten(parameters)
    flat_batched, _ = tree_flatten(batched)
    flat_batched_params, flat_static_params = zip(*[(p, None) if b else (None, p)
                                                    for p, b in zip(flat_params, flat_batched)])

    batched_params = tree_unflatten(tree_def, flat_batched_params)
    static_params = tree_unflatten(tree_def, flat_static_params)
    return batched_params, static_params


@partial(jax.jit, static_argnums=(2,))
def rejoin_batched_and_static_params(batched_params, static_params, batched):
    flat_batched, _ = tree_flatten(batched)

    # By default jax tree_flatten discards the None, so we need to prevent it
    # from doing so by say that the None is indeed a leaf.
    flat_batched_params, tree_def = tree_flatten(batched_params, is_leaf=lambda z: z is None)
    flat_static_params, _ = tree_flatten(static_params, is_leaf=lambda z: z is None)

    flat_params = [p1 if b else p2 for (p1, p2, b) in zip(flat_batched_params, flat_static_params, flat_batched)]
    return tree_unflatten(tree_def, flat_params)


class ParametrizedModel:
    def __init__(self, parameters: PyTree, batched: PyTree):
        self.parameters = parameters
        self.batched = batched


class UnivariatePotentialModel(ParametrizedModel):
    r"""
    Univariate potential model. For example this can represent x -> log(p(y|x)) or a given density, in which case the
    log-potential corresponds to the unnormalised log-likelihood.
    It is used to define M_0, G_0, and :math:`\nu_t`.

    Parameters
    ----------
    parameters: PyTree
        The parameters of the potential model for each time step.
    batched: PyTree
        Are the parameters batched along the first dimension (where each index corresponds to a time step) or is
        parameters constant over all time steps. The tree structure is the same as parameters.
    """

    @classmethod
    def log_potential(cls, particle: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        """
        This computes the log-potential of a given particle (for a given time step)
        under the proposal model. This will be vmapped.

        Parameters
        ----------
        particle: chex.ArrayTree
            Current particle to be evaluated
        parameter: PyTree
            The parameter of the proposal distribution at the current time step.

        Returns
        -------
        res: jnp.ndarray
            Resulting log-potentials of the samples in the model
        """
        raise NotImplementedError


class DensityModel(UnivariatePotentialModel):
    """
    Same as the univariate potential model, but can also sample from the model. In practice this should be reserved to
    :math:`q_t` even if the potential model comes from a density in the first place (coding best practices).

    Parameters
    ----------
    parameters: PyTree
        The parameters of the proposal distribution for each time step.
    batched: PyTree
        Are the parameters batched along the first dimension (where each index corresponds to a time step) or is
        parameters constant over all time steps. The tree structure is the same as parameters.
    T: int
        Number of time steps sampled from.
    """

    def __init__(self, parameters: PyTree, batched: PyTree, T: int = None):
        ParametrizedModel.__init__(self, parameters, batched)
        self.T = T

    def sample(self, key: chex.PRNGKey, N: int) -> chex.ArrayTree:
        """
        This samples the full proposal trajectories for the model.
        If `T` is given, then we consider that the same parameters are used at each time step,
        otherwise we consider that parameters leaves are batched in the first dimension.

        Parameters
        ----------
        key: chex.PRNGKey
            The jax random key used for sampling
        N: int
            Number of trajectories per time step

        Returns
        -------
        trajectories: chex.ArrayTree
            Resulting sampled trajectories. First batch index is the time index,
            second is the simulation. The trailing dimension are model specific.
        """
        raise NotImplementedError("")


class BivariatePotentialModel(ParametrizedModel):
    r"""
    Bivariate potential model. For example this can represent x_t_1, x_t -> log(p(x_t|x_t_1)).
    It is used to define M_t and G_t.

    Parameters
    ----------
    parameters: PyTree
        The parameters of the potential model for each time step.
    batched: PyTree
        Are the parameters batched along the first dimension (where each index corresponds to a time step) or is
        parameters constant over all time steps. The tree structure is the same as parameters.
    """

    @classmethod
    def log_potential(cls, x_t_1: chex.ArrayTree, x_t: chex.ArrayTree, parameter: PyTree) -> jnp.ndarray:
        """
        This computes the unnormalised log-potential of a given pair of particles (for a given time step)
        under the model. This will be vmapped.

        Parameters
        ----------
        x_t_1: chex.ArrayTree
            Particle at time t-1
        x_t: chex.ArrayTree
            Particle at time t
        parameter: PyTree
            The parameter of the proposal distribution at time t.

        Returns
        -------
        res: jnp.ndarray
            Resulting log-potentials of the samples in the model
        """
        raise NotImplementedError


class ConditionalDensityModel(BivariatePotentialModel):
    r"""
    Conditional Density Model, same as BivariatePotentialModel with additional sampling utility.

    Parameters
    ----------
    parameters: PyTree
        The parameters of the potential model for each time step. Note that it corresponds to the time step for `t`.
    batched: PyTree
            Are the parameters batched along the first dimension (where each index corresponds to a time step) or is
            parameters constant over all time steps. The tree structure is the same as parameters.
    """

    @classmethod
    def sample(cls, key: chex.PRNGKey, x_t_1: chex.ArrayTree, parameter: PyTree) -> chex.ArrayTree:
        """
        This samples X_t conditonally on x_t_1. This is batched across the simulation.

        Parameters
        ----------
        key: chex.PRNGKey
            The jax random key used for sampling
        x_t_1: chex.ArrayTree
            Particles at time t-1
        parameter: PyTree
            The parameter of the proposal distribution at time t.

        Returns
        -------
        x_t: chex.ArrayTree
            Proposed particles at time t
        """
        raise NotImplementedError


class NullPotentialModel(UnivariatePotentialModel):
    """
    Null potential function
    Corresponds to an uninformative (or absent) information.
    """
    def __init__(self):
        UnivariatePotentialModel.__init__(self, None, None)

    @classmethod
    def log_potential(cls, particles: chex.ArrayTree, parameter: PyTree) -> float:
        return 0.
