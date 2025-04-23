"""
This module emulates one of the examples from the primary reference associated with this project,
namely,

Elsayed, M., Vasan, G., & Mahmood, A. R. (2024). Streaming Deep Reinforcement Learning Finally Works. 
arXiv preprint arXiv:2410.14606.

Here, we use a Generalized Value Function to predict future oil temperatures in the electricity transformer
temperature dataset, which is described in,

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021, May).
Informer: Beyond efficient transformer for long sequence time-series forecasting.
In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 12, pp. 11106-11115).
"""

import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from data.data import prepare_data
from streamx.scalers import scale_reward, normalize_observations
from streamx.optimizer import overshoot_bounded_gd
from streamx.model import Model

## Load data
feature_data, target_data = prepare_data(skip=0)

# Plot signals/variables 
fig, ax = plt.subplots(figsize=(8,5))

col_names = feature_data.columns.to_list()[:-1]
for c in col_names:
    ax.plot(feature_data[c].to_numpy(), label=f'{c}')
ax.plot(target_data.to_numpy(), label='Oil Temp.')
ax.legend()

ax.set_title('Input and Output Signals')
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
plt.savefig('io_signals.png')
plt.close()

## Example hyperparameter settings

# experiment settings
discount_factor = 0.99 # prediction horizion of 100 time steps (1/(1-gamma))
lambd = 0.8 # eligibility trace parameter, lambd in the TD(lambd) algorithm

# model settings
in_features = feature_data.shape[1]
hidden_features = 128
out_features = 1
sparsity_level = 0.9

# optimization stettings
scaling_factor = 2
learning_rate = 1

# dataset
n_data = feature_data.shape[0]

# observation memory traces
state_decay = 0.999  # decay factor for state in observation trace (exp. weighted moving average)

# observation state normalization
state_mean = 0
state_unvar = 1

# reward scaling
reward_state = 0
reward_unvar = 1

## Model

# The following neural network will be used as a generalized value function for predicting the return.
model = Model(input_dim=in_features,
              hidden_dim=hidden_features,
              output_dim=out_features,
              sparsity_level=sparsity_level,
              rngs=nnx.Rngs(0))

# initialize optimizer
tx = overshoot_bounded_gd(learning_rate,
                          lambd * discount_factor,
                          scaling_factor)

optimizer = nnx.Optimizer(model, tx)

# initial observation
current_state = jnp.array(feature_data.iloc[0])

# inital observation state
outs = normalize_observations(current_state,
                              state_mean,
                              state_unvar,
                              0)

current_normalized_state = outs[0]
state_mean = outs[1]
state_unvar = outs[2] 
state_std = outs[3]

## metrics
compute_metrics_freq = 10 # matplotlib can cause errors on macs if too much data is plotted

# true return
T = 1000 # future steps to approximate the discounted infinite horizon
discounts = discount_factor**(jnp.arange(T))
G = []
V = []

@nnx.jit
def update_step(model, optimizer, x, y):
    def model_eval(net: nnx.Module):
        return net(x)[0]
    
    grads = nnx.grad(model_eval)(model)
    optimizer.update(grads, td_error=y)

for step in tqdm(range(1, n_data - T)):

    # update memory traces with most recent observations
    next_obs = jnp.array(feature_data.iloc[step])
    next_state = state_decay * current_state + (1-state_decay)*next_obs
    next_reward = jnp.array(target_data.iloc[step])

    # normalize observations
    outs = normalize_observations(next_state,
                                  state_mean,
                                  state_unvar,
                                  step)

    next_normalized_state = outs[0]
    state_mean = outs[1]
    state_unvar = outs[2] 
    state_sigma = outs[3]

    # scale rewards
    routs = scale_reward(reward_state,
                         next_reward,
                         discount_factor,
                         reward_unvar,
                         step)
    scaled_reward = routs[0]
    reward_state = routs[1]
    reward_unvar = routs[2] 
    reward_sigma = routs[3]
    
    # compute TD error
    td_error = scaled_reward + discount_factor * model(next_normalized_state) - model(current_normalized_state)

    # update value/prediction function weights
    update_step(model, optimizer, current_normalized_state, td_error)

    # increment memory traces and states
    current_state = next_state
    current_normalized_state = next_normalized_state

    ## metrics
    if step % compute_metrics_freq == 0:
        # true return
        G.append(jnp.inner(discounts, jnp.array(target_data.iloc[step+1:step+T+1])))
        # predicted return
        V.append(reward_sigma * model(current_normalized_state)) #rescaled GVF prediction

## Plot results

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(G, 'r', label='Ground truth')
ax.plot(V, 'b--', label='Predicted via GVF')
ax.set_xlabel('Time Step')
ax.set_ylabel('Return')
ax.set_title('Return (discounted oil temperature prediction)')
ax.legend()
plt.savefig('return_predictions.png')
plt.close()

  