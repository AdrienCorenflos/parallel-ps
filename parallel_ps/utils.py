import math

import jax.numpy as jnp
import jax.scipy


def mvn_loglikelihood(x, mean, chol_cov, is_diag=False):
    """multivariate normal"""
    dim = x.shape[0]
    if is_diag:
        y = (x - mean) / chol_cov
        normalizing_constant = jnp.sum(jnp.log(jnp.abs(chol_cov)))
    else:
        y = jax.scipy.linalg.solve_triangular(chol_cov, x - mean, lower=True)
        normalizing_constant = jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_cov))))

    norm_y = jnp.sum(y * y, -1)
    return -0.5 * norm_y - normalizing_constant - dim * math.log(2 * math.pi) / 2.0
