"""
This module contains functionality for scaling and normalization.
"""

import jax
import jax.numpy as jnp

def sample_mean_var(current_val: float | jax.Array,
                    current_mean: float | jax.Array,
                    current_unvar: float | jax.Array,
                    current_iterate: int) -> tuple[float | jax.Array,
                                                   float | jax.Array,
                                                   float | jax.Array,
                                                   int]:
    """
    Recursive estimates of the input sequence mean and variance.

    Args:
        current_val (float | jax.Array): current input
        current_mean (float | jax.Array): current mean estimate of x
        current_unvar (float | jax.Array): current unnormalized variance 
        current_iterate (int): current iterate

    Returns:
        (float | jax.Array): updated mean estimate
        (float | jax.Array): updated unnormalized variance
        (float | jax.Array): updated variance
        (int): incremented iterate
    """

    next_iterate = current_iterate+1
    new_mean = current_mean + 1/next_iterate*(current_val - current_mean)
    new_unvar= current_unvar + (current_val - current_mean) * (current_val - new_mean)
    sigma2 = new_unvar/(next_iterate-1) if next_iterate >= 2 else 1

    return new_mean, new_unvar, sigma2, next_iterate


def normalize_observations(current_state: float | jax.Array,
                           current_mean: float | jax.Array,
                           current_unvar: float | jax.Array,
                           current_iterate: int,
                           eps: float = 1e-7,
                           ) -> tuple[float | jax.Array,
                                      float | jax.Array,
                                      float | jax.Array]:
    
    new_mean, new_unvar, sigma2, next_iterate = sample_mean_var(
            current_val=current_state,
            current_mean=current_mean,
            current_unvar=current_unvar,
            current_iterate=current_iterate,
        )
    
    sigma = jnp.sqrt(sigma2 + eps)
    normalized_state = (current_state - new_mean) / sigma

    
    return normalized_state, new_mean, new_unvar, sigma
    


def scale_reward(reward_state: float | jax.Array,
                 new_reward: float | jax.Array,
                 discount_factor: float | jax.Array,
                 current_unvar: float | jax.Array,
                 current_iterate: int,
                 eps: float = 1e-7) -> tuple[float | jax.Array,
                                             float | jax.Array,
                                             float | jax.Array]:
    
    reward_state = discount_factor * reward_state + new_reward 
        
    new_mean, new_unvar, sigma2, next_iterate = sample_mean_var(
        current_val=reward_state,
        current_mean=0,
        current_unvar=current_unvar,
        current_iterate=current_iterate,
    )

    sigma = jnp.sqrt(sigma2 + eps)
    return new_reward/sigma, reward_state, new_unvar, sigma

    

        