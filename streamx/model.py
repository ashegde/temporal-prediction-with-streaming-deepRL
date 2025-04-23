"""
This module contains functionality for neural network models.
"""
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

def sparse_init(sparsity_level: float = 0.9) -> jax.nn.initializers.Initializer:
    """
    Builds an initializer that returns a LeCun initialized kernel,
    postprocessed to have the target sparsity level.

    Based on Algorithm 1 of the primary reference.

    Args:
        sparsity_level (float): proportion of zero-valued weights 
        dtype (Any): default dtype of the weights

    Returns:
        (jax.nn.initializers.Intializer): An initializer object
    """


    def init(key: jax.Array,
             shape: Tuple,
             dtype: Any = jnp.float_) -> jax.Array:
        fan_in = shape[0]
        fan_out = shape[1]
        scale = 1 / jnp.sqrt(fan_in)
        W = jax.random.uniform(key, shape=(fan_in, fan_out),
                               minval=-scale, maxval=scale, dtype=dtype)
        nz = jnp.floor(sparsity_level * fan_in).astype(int)
        perm_idx = jax.random.permutation(key, fan_in)
        return W.at[perm_idx[:nz], :].set(0) 
        # return W.at[:, perm_idx[:nz]].set(0) 
    
    return init


def layer_norm(a: jnp.array, eps: float = 1e-7) -> jnp.array:
    """
    LayerNorm without any trainable parameters 

    Args:
        a (jnp.array): incoming activations
        eps (float): tolerance

    Returns:
        (jnp.array): normalized activations
    """
    a_mean = jnp.mean(a, axis=-1)
    a_var = jnp.var(a, axis=-1, keepdims=True)
    return (a - a_mean) / jnp.sqrt(a_var + eps)


class Model(nnx.Module):
    """
    MLP with two hidden layers and layer norms.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, sparsity_level: float, rngs: nnx.Rngs):
        self.linear_in = nnx.Linear(input_dim, hidden_dim, use_bias=True,
                                    kernel_init=sparse_init(sparsity_level), rngs=rngs)
        self.linear_mid = nnx.Linear(hidden_dim, hidden_dim, use_bias=True,
                                     kernel_init=sparse_init(sparsity_level), rngs=rngs)
        self.linear_out = nnx.Linear(hidden_dim, output_dim, use_bias=True,
                                     kernel_init=sparse_init(sparsity_level), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.leaky_relu(layer_norm(self.linear_in(x)))
        h = nnx.leaky_relu(layer_norm(self.linear_mid(h)))
        return self.linear_out(h) 

        