# %%[markdown]
# #### Neural Fitted Q-Iteration with Continuous Actions (NFQCA)

# **Input** MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B}$, Q-function $q(s,a; \boldsymbol{\theta})$, policy $d(s; \boldsymbol{w})$

# **Output** Parameters $\boldsymbol{\theta}$ for Q-function, $\boldsymbol{w}$ for policy

# 1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
# 2. **for** $n = 0,1,2,...$ **do**
#     1. $\mathcal{D}_q \leftarrow \emptyset$
#     2. For each $(s,a,r,s') \in \mathcal{B}$:
#         1. $a'_{s'} \leftarrow d(s'; \boldsymbol{w}_n)$
#         2. $y_{s,a} \leftarrow r + \gamma q(s', a'_{s'}; \boldsymbol{\theta}_n)$
#         3. $\mathcal{D}_q \leftarrow \mathcal{D}_q \cup \{((s,a), y_{s,a})\}$
#     3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_q)$
#     4. $\boldsymbol{w}_{n+1} \leftarrow \texttt{minimize}_{\boldsymbol{w}} -\frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s') \in \mathcal{B}} q(s, d(s; \boldsymbol{w}); \boldsymbol{\theta}_{n+1})$
# 3. **return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
# %%
import os
import warnings
import pickle as pkl
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from functools import partial
from matplotlib import pyplot as plt

# Import your configuration
from config import SimulationParams

# Suppress warnings and set working directory
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%

da_path = "Results/offline_dataset_day_ahead.pkl"
id_path = "Results/offline_dataset_intraday.pkl"

da_df = pkl.load(open(da_path, "rb"))
id_df = pkl.load(open(id_path, "rb"))

# Load Simulation Parameters
sim_params = SimulationParams()

# %%

# print shapes:
print("Day-ahead data:")
for key in da_df.keys():
    print(f"Day Ahead : {key}, {da_df[key][2].shape}")
# %%
print("Intraday data:")
for key in id_df.keys():
    print(f"Intraday : {key}, {id_df[key][2].shape}")
# %%

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from functools import partial
import numpy as np
import pickle

# ------------------------
# Parameters
# ------------------------
# Season = "Summer"
# length_R = 5
# seed = 2
# D = 7  # days in forecast
# Rmax = 100
# np.random.seed(seed)

# t_ramp_pump_up = 2 / 60
# t_ramp_pump_down = 2 / 60
# t_ramp_turbine_up = 2 / 60
# t_ramp_turbine_down = 2 / 60

# c_grid_fee = 5 / 4
# Delta_ti = 0.25
# Delta_td = 1.0

# Q_mult = 1.2
# Q_fix = 3
# Q_start_pump = 15
# Q_start_turbine = 15

# beta_pump = 0.9
# beta_turbine = 0.9

# x_max_pump = 10
# x_min_pump = 5
# x_max_turbine = 10
# x_min_turbine = 5

# R_vec = np.linspace(0, Rmax, length_R)
# x_vec = np.array([-x_max_turbine, 0, x_max_pump])

# c_pump_up = t_ramp_pump_up / 2
# c_pump_down = t_ramp_pump_down / 2
# c_turbine_up = t_ramp_turbine_up / 2
# c_turbine_down = t_ramp_turbine_down / 2


# ------------------------
# Load Data
# ------------------------
def load_offline_data_da(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    states = np.stack(df["state"].values)
    actions = np.stack(df["action"].values)  # now treated as continuous
    rewards = df["reward"].values
    next_states = np.stack(df["next_state"].values)
    return states, actions, rewards, next_states


def load_offline_data_id(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    states = np.stack(df["state"].values)
    actions = np.stack(df["action"].values)  # now treated as continuous
    rewards = df["reward"].values
    next_states = np.stack(df["next_state"].values)
    return states, actions, rewards, next_states


def batch_iter(data, batch_size, shuffle=True):
    states, actions, rewards, next_states = data
    N = len(states)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        idx = indices[start : start + batch_size]
        yield (states[idx], actions[idx], rewards[idx], next_states[idx])


# ------------------------
# Model Definitions
# ------------------------
class QNetworkDA(nn.Module):
    state_dim: int = 842
    action_dim: int = 24
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state, action):
        # action is continuous now, just ensure float32
        x = jnp.concatenate([state, action.astype(jnp.float32)], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class QNetworkID(nn.Module):
    state_dim: int = 890
    action_dim: int = 1152
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action.astype(jnp.float32)], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class PolicyDA(nn.Module):
    ub : float # upper bound
    lb : float # lower bound
    state_dim: int = 842
    action_dim: int = 24
    hidden_dim: int = 256


    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        # Output a vector of action_dim floats (continuous actions)
        actions = nn.Dense(self.action_dim)(x)

        actions = self.lb + (self.ub - self.lb) * nn.sigmoid(actions)

        return actions
    
# Define large finite values to approximate infinity
NEG_INF = -1e6
POS_INF = 1e6

# Define bounds for PolicyID
lb_id = np.concatenate([
    np.zeros(96),               # First 96 actions: lower bound = 0
    NEG_INF * np.ones(672),     # Next 672 actions: lower bound = -1e6 (approx. -inf)
    np.zeros(384)               # Last 384 actions: lower bound = 0
]).astype(np.float32)

ub_id = np.concatenate([
    sim_params.x_max_pump * np.ones(96),  # First 96 actions: upper bound = Rmax
    POS_INF * np.ones(672),              # Next 672 actions: upper bound = 1e6 (approx. inf)
    np.ones(384)                         # Last 384 actions: upper bound = 1
]).astype(np.float32)



class PolicyID(nn.Module):
    lb: jnp.ndarray  # Lower bounds, shape (1152,)
    ub: jnp.ndarray  # Upper bounds, shape (1152,)
    state_dim: int = 890
    action_dim: int = 1152
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        # Forward pass through the network
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Output raw actions
        raw_actions = nn.Dense(self.action_dim)(x)
        
        # Define masks for bounded and unbounded actions
        # Bounded: first 96 and last 384 actions
        # Unbounded: middle 672 actions
        mask_bound = jnp.concatenate([
            jnp.ones(96),
            jnp.zeros(672),
            jnp.ones(384)
        ])
        
        # Apply sigmoid scaling to bounded actions
        scaled_actions = self.lb + (self.ub - self.lb) * nn.sigmoid(raw_actions)
        
        # Combine scaled and unscaled actions
        final_actions = mask_bound * scaled_actions + (1.0 - mask_bound) * raw_actions
        
        return final_actions



# Initialize PRNGKeys
key = jax.random.PRNGKey(0)
da_key, id_key, pda_key, pid_key = jax.random.split(key, 4)

# Dummy inputs for initialization (continuous actions)
dummy_s_da = jnp.ones((1, 842))
dummy_a_da = jnp.ones((1, 24), dtype=jnp.float32)
dummy_s_id = jnp.ones((1, 890))
dummy_a_id = jnp.ones((1, 1152), dtype=jnp.float32)

q_da_model = QNetworkDA()
q_id_model = QNetworkID()
policy_da_model = PolicyDA(ub= sim_params.x_max_pump, lb= -1*sim_params.x_max_turbine)
# Convert lb_id and ub_id to JAX arrays
lb_id_jax = jnp.array(lb_id)
ub_id_jax = jnp.array(ub_id)
# Initialize PolicyID with lb and ub
policy_id_model = PolicyID(lb=lb_id_jax, ub=ub_id_jax)


q_da_params = q_da_model.init(da_key, dummy_s_da, dummy_a_da)
q_id_params = q_id_model.init(id_key, dummy_s_id, dummy_a_id)
policy_da_params = policy_da_model.init(pda_key, dummy_s_da)
policy_id_params = policy_id_model.init(pid_key, dummy_s_id)

q_da_target_params = q_da_params
q_id_target_params = q_id_params

learning_rate = 1e-3
q_da_opt = optax.adam(learning_rate)
q_id_opt = optax.adam(learning_rate)
policy_da_opt = optax.adam(learning_rate)
policy_id_opt = optax.adam(learning_rate)

q_da_opt_state = q_da_opt.init(q_da_params)
q_id_opt_state = q_id_opt.init(q_id_params)
policy_da_opt_state = policy_da_opt.init(policy_da_params)
policy_id_opt_state = policy_id_opt.init(policy_id_params)

gamma = 0.99
batch_size = 64
num_epochs = 10

da_data = load_offline_data_da("Results/offline_dataset_day_ahead.pkl")
id_data = load_offline_data_id("Results/offline_dataset_intraday.pkl")


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def soft_update(target_params, online_params, tau=0.005):
    return jax.tree_util.tree_map(
        lambda tp, op: tp * (1 - tau) + op * tau, target_params, online_params
    )


@jax.jit
def update_q_id(
    q_id_params,
    q_id_opt_state,
    q_id_target_params,
    q_da_target_params,
    policy_da_params,
    s_id,
    a_id,
    r_id,
    s_da_next,
):
    # Q_ID target: R_t^{ID} + gamma * Q_DA(s_{t+1}^{DA}, policy_DA(s_{t+1}^{DA}))
    next_da_actions = policy_da_model.apply(policy_da_params, s_da_next)
    q_da_values = q_da_model.apply(q_da_target_params, s_da_next, next_da_actions)
    q_target_id = r_id + gamma * q_da_values

    def loss_fn(params):
        q_estimate = q_id_model.apply(params, s_id, a_id)
        return mse_loss(q_estimate, q_target_id), q_estimate

    grads, (q_estimate) = jax.grad(loss_fn, has_aux=True)(q_id_params)
    updates, q_id_opt_state_new = q_id_opt.update(grads, q_id_opt_state)
    q_id_params_new = optax.apply_updates(q_id_params, updates)
    return q_id_params_new, q_id_opt_state_new, q_estimate


@jax.jit
def update_q_da(
    q_da_params,
    q_da_opt_state,
    q_da_target_params,
    q_id_target_params,
    policy_id_params,
    s_da,
    a_da,
    r_da,
    s_id_next,
):
    # Q_DA target: R_t^{DA} + gamma * Q_ID(s_{t}^{ID}, policy_ID(s_{t}^{ID}))
    next_id_actions = policy_id_model.apply(policy_id_params, s_id_next)
    q_id_values = q_id_model.apply(q_id_target_params, s_id_next, next_id_actions)
    q_target_da = r_da + gamma * q_id_values

    def loss_fn(params):
        q_da_values = q_da_model.apply(params, s_da, a_da)
        loss = mse_loss(q_da_values, q_target_da)
        return loss, q_da_values

    grads, (q_da_values) = jax.grad(loss_fn, has_aux=True)(q_da_params)
    updates, q_da_opt_state_new = q_da_opt.update(grads, q_da_opt_state)
    q_da_params_new = optax.apply_updates(q_da_params, updates)
    return q_da_params_new, q_da_opt_state_new, q_da_values


@jax.jit
def update_policy_da(policy_da_params, policy_da_opt_state, q_da_params, s_da):
    # Deterministic policy gradient: maximize Q(s, pi(s))
    # loss = -mean(Q(s, pi(s)))
    def loss_fn(params):
        a_da = policy_da_model.apply(params, s_da)  # continuous actions
        q_values = q_da_model.apply(q_da_params, s_da, a_da)
        return -jnp.mean(q_values)

    grads = jax.grad(loss_fn)(policy_da_params)
    updates, policy_da_opt_state_new = policy_da_opt.update(grads, policy_da_opt_state)
    policy_da_params_new = optax.apply_updates(policy_da_params, updates)
    return policy_da_params_new, policy_da_opt_state_new


@jax.jit
def update_policy_id(policy_id_params, policy_id_opt_state, q_id_params, s_id):
    # Similarly for ID: maximize Q(s, pi(s))
    def loss_fn(params):
        a_id = policy_id_model.apply(params, s_id)
        q_values = q_id_model.apply(q_id_params, s_id, a_id)
        return -jnp.mean(q_values)

    grads = jax.grad(loss_fn)(policy_id_params)
    updates, policy_id_opt_state_new = policy_id_opt.update(grads, policy_id_opt_state)
    policy_id_params_new = optax.apply_updates(policy_id_params, updates)
    return policy_id_params_new, policy_id_opt_state_new


# ------------------------
# Training Loop
# ------------------------
for epoch in range(num_epochs):
    # Train Q_ID and then Policy_ID
    for s_id, a_id, r_id, s_da_next in batch_iter(id_data, batch_size, shuffle=True):
        s_id = jnp.array(s_id, dtype=jnp.float32)
        a_id = jnp.array(a_id, dtype=jnp.float32)  # continuous now
        r_id = jnp.array(r_id, dtype=jnp.float32).reshape(-1, 1)
        s_da_next = jnp.array(s_da_next, dtype=jnp.float32)

        q_id_params, q_id_opt_state, q_est_id = update_q_id(
            q_id_params,
            q_id_opt_state,
            q_id_target_params,
            q_da_target_params,
            policy_da_params,
            s_id,
            a_id,
            r_id,
            s_da_next,
        )
        # Update ID policy: now just maximize Q(s, pi(s))
        policy_id_params, policy_id_opt_state = update_policy_id(
            policy_id_params, policy_id_opt_state, q_id_params, s_id
        )

    # Train Q_DA and then Policy_DA
    for s_da, a_da, r_da, s_id_next in batch_iter(da_data, batch_size, shuffle=True):
        s_da = jnp.array(s_da, dtype=jnp.float32)
        a_da = jnp.array(a_da, dtype=jnp.float32)
        r_da = jnp.array(r_da, dtype=jnp.float32).reshape(-1, 1)
        s_id_next = jnp.array(s_id_next, dtype=jnp.float32)

        q_da_params, q_da_opt_state, q_est_da = update_q_da(
            q_da_params,
            q_da_opt_state,
            q_da_target_params,
            q_id_target_params,
            policy_id_params,
            s_da,
            a_da,
            r_da,
            s_id_next,
        )
        # Update DA policy
        policy_da_params, policy_da_opt_state = update_policy_da(
            policy_da_params, policy_da_opt_state, q_da_params, s_da
        )

    q_da_target_params = soft_update(q_da_target_params, q_da_params)
    q_id_target_params = soft_update(q_id_target_params, q_id_params)

    print(f"Epoch {epoch+1}/{num_epochs} finished.")


# %%
# ------------------------
# Using the Policy
# ------------------------
def sample_action_da(policy_da_params, s_da_example):
    # now returns continuous action vector
    actions = policy_da_model.apply(policy_da_params, s_da_example)
    return actions


def sample_action_id(policy_id_params, s_id_example):
    # returns continuous action vector
    actions = policy_id_model.apply(policy_id_params, s_id_example)
    return actions


# %%
# Example usage:
s_da_example = jnp.ones((1, 842), dtype=jnp.float32)
da_action = sample_action_da(policy_da_params, s_da_example)
s_id_example = jnp.ones((1, 890), dtype=jnp.float32)
id_action = sample_action_id(policy_id_params, s_id_example)

print("Selected DA action (continuous):", da_action)
print("Selected ID action (continuous):", id_action)

# %%
