"""
This module contains functionality for the Overshooting-bounded Gradient Descent algorithm
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from optax._src import base
from optax._src import combine


class TraceState(NamedTuple):
    """State for Eligibility Trace"""
    z: base.Updates # eligibility trace


def trace_update(update: Any, state: Any, decay: float) -> Any:
    """Compute an update of the trace"""
    return jax.tree.map(
        lambda g, s: decay * s + g if g is not None else None,
        update,
        state,
        is_leaf=lambda x: x is None,
    )


def scale_by_obgd_lr(learning_rate: base.ScalarOrSchedule,
                     trace_decay: float,
                     scaling_factor: float) -> base.GradientTransformationExtraArgs:
    
    """
    Modifies and scales the update according to the Overshooting-bounded Gradient Descent algorithm.

    Args:
        learning_rate (base.ScalarOrSchedule): learning rate
        td_error (float): temporal difference error
        trace_decay (float): coefficient for the eligibility trace
        scaling_factor (float): scaling factor for step size bound

    Returns:
        (base.GradientTranformation): A gradient transformation object
    """

    def init_fn(params: base.Params):
        #initialize the gradient eligibility trace to zero
        return TraceState(z=otu.tree_zeros_like(params)) 

    def update_fn(updates: base.Updates, state: TraceState, params: base.Params = None, td_error: float = 0):
        # grads = updates
        
        # Eligibility trace update
        z = trace_update(updates, state.z, trace_decay)

        # Compute learning rate
        delta_bar = jnp.maximum(jnp.abs(td_error), 1)
        M = scaling_factor * delta_bar * otu.tree_l1_norm(z)
        lr = jnp.minimum(1/M, learning_rate)

        # Updated state
        updates = jax.tree.map(
            lambda g: lr * td_error * g, 
            z,
        )
        return updates, TraceState(z=z)
    
    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def overshoot_bounded_gd(learning_rate: base.ScalarOrSchedule,
                         trace_decay: float,
                         scaling_factor: float) -> base.GradientTransformation:
    """
    Overshooting-bounded Gradient Descent (OBGD)

    This function implements Algorithm 3 of the primary reference,
    which controls the effective step size of each update.

    Note that the ``init`` function of this optimizer creates an internal
    state, which contains the eligibility trace of the incoming value gradient.
    These values are initialized and stored as pytrees containing all zeros,
    with the same shape as the model updates.

    Args:
        learning_rate (base.ScalarOrSchedule): learning rate
        td_error (float): temporal difference error
        trace_decay (float): coefficient for the eligibility trace
        scaling_factor (float): scaling factor for step size bound

    Returns:
        (base.GradientTransformation): gradient transformation for the OBDG algorithm

    """

    return combine.chain(
        scale_by_obgd_lr(learning_rate, trace_decay, scaling_factor),
    )
    

