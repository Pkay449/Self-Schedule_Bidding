# %%
# %%
import os
import warnings
import pickle
from dataclasses import dataclass
from typing import Any, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jaxopt import OSQP
from matplotlib import pyplot as plt

import helper as h_

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
# ------------------------
# Configuration
# ------------------------
@dataclass
class Config:
    # General Parameters
    season: str = "Summer"
    length_R: int = 5
    seed: int = 2
    D: int = 7  # days in forecast
    Rmax: float = 100.0
    gamma: float = 0.99
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3

    # Ramp Times (in hours)
    t_ramp_pump_up: float = 2 / 60
    t_ramp_pump_down: float = 2 / 60
    t_ramp_turbine_up: float = 2 / 60
    t_ramp_turbine_down: float = 2 / 60

    # Costs and Quantities
    c_grid_fee: float = 5 / 4
    Delta_ti: float = 0.25
    Delta_td: float = 1.0
    Q_mult: float = 1.2
    Q_fix: float = 3.0
    Q_start_pump: float = 15.0
    Q_start_turbine: float = 15.0

    # Efficiency
    beta_pump: float = 0.9
    beta_turbine: float = 0.9

    # Action Bounds
    x_max_pump: float = 10.0
    x_min_pump: float = 5.0
    x_max_turbine: float = 10.0
    x_min_turbine: float = 5.0

    # Derived Parameters
    R_vec: np.ndarray = np.linspace(0, 100, 5)
    x_vec: np.ndarray = np.array([-10, 0, 10])

    # Ramp Costs
    c_pump_up: float = t_ramp_pump_up / 2
    c_pump_down: float = t_ramp_pump_down / 2
    c_turbine_up: float = t_ramp_turbine_up / 2
    c_turbine_down: float = t_ramp_turbine_down / 2

config = Config()
np.random.seed(config.seed)

# %%
# ------------------------
# Data Loading
# ------------------------
def load_offline_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        df = pickle.load(f)
    states = np.stack(df["state"].values)
    actions = np.stack(df["action"].values).astype(np.float32)  # treated as continuous
    rewards = df["reward"].values.astype(np.float32)
    next_states = np.stack(df["next_state"].values)
    return states, actions, rewards, next_states

def print_data_shapes(data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], data_type: str) -> None:
    states, actions, rewards, next_states = data
    print(f"{data_type} data:")
    print(f"{data_type} States Shape: {states.shape}")
    print(f"{data_type} Actions Shape: {actions.shape}")
    print(f"{data_type} Rewards Shape: {rewards.shape}")
    print(f"{data_type} Next States Shape: {next_states.shape}")

da_path = "Results/offline_dataset_day_ahead.pkl"
id_path = "Results/offline_dataset_intraday.pkl"

# Load data using the defined function
da_data = load_offline_data(da_path)
id_data = load_offline_data(id_path)

# Print data shapes for verification
print_data_shapes(da_data, "Day Ahead")
print_data_shapes(id_data, "Intraday")

# %%
# ------------------------
# Batch Iterator
# ------------------------
def batch_iter(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states, actions, rewards, next_states = data
    N = len(states)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        idx = indices[start : start + batch_size]
        yield states[idx], actions[idx], rewards[idx], next_states[idx]

# %%
# ------------------------
# Model Definitions
# ------------------------
class QNetwork(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)

class PolicyNetworkDA(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray, ub: jnp.ndarray, lb: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        actions = nn.Dense(self.action_dim)(x)
        return lb + (ub - lb) * nn.sigmoid(actions)

class PolicyNetworkID(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        Aeq: List[jnp.ndarray],
        beq: List[jnp.ndarray],
        A: List[jnp.ndarray],
        b: List[jnp.ndarray],
        ub: jnp.ndarray,
        lb: jnp.ndarray
    ) -> jnp.ndarray:
        # Generate raw actions
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        raw_actions = nn.Dense(self.action_dim)(x)

        # Combine constraints
        Aeq_all, beq_all = combine_constraints(Aeq, beq, equality=True)
        A_all, b_all = combine_constraints(A, b, equality=False)

        # Define QP problem
        actions_feasible = solve_qp(
            raw_actions,
            Aeq_all,
            beq_all,
            A_all,
            b_all,
            ub,
            lb,
            self.action_dim
        )
        return actions_feasible

def combine_constraints(A_constraints: List[jnp.ndarray], b_constraints: List[jnp.ndarray], equality: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if A_constraints:
        A_combined = jnp.vstack(A_constraints)
        b_combined = jnp.concatenate(b_constraints)
    else:
        A_combined = jnp.zeros((0, A_constraints[0].shape[-1] if A_constraints else 0))
        b_combined = jnp.zeros((0,))
    return A_combined, b_combined

def solve_qp(
    raw_actions: jnp.ndarray,
    Aeq: jnp.ndarray,
    beq: jnp.ndarray,
    A: jnp.ndarray,
    b: jnp.ndarray,
    ub: jnp.ndarray,
    lb: jnp.ndarray,
    action_dim: int
) -> jnp.ndarray:
    # Objective: minimize (1/2)*x^T x - raw_actions^T x
    P = jnp.eye(action_dim)
    q = -raw_actions

    # Equality constraints
    if Aeq.shape[0] > 0:
        A_eq = Aeq
        l_eq = beq
        u_eq = beq
    else:
        A_eq = jnp.zeros((0, action_dim))
        l_eq = jnp.zeros((0,))
        u_eq = jnp.zeros((0,))

    # Inequality constraints
    if A.shape[0] > 0:
        A_ineq = A
        l_ineq = -jnp.inf * jnp.ones(A.shape[0])
        u_ineq = b
    else:
        A_ineq = jnp.zeros((0, action_dim))
        l_ineq = jnp.zeros((0,))
        u_ineq = jnp.zeros((0,))

    # Bounds as linear constraints
    A_bounds = jnp.vstack([jnp.eye(action_dim), -jnp.eye(action_dim)])
    l_bounds = jnp.concatenate([lb, -ub])
    u_bounds = jnp.concatenate([ub, -lb])

    # Combine all constraints
    A_total = jnp.vstack([A_eq, A_ineq, A_bounds])
    l_total = jnp.concatenate([l_eq, l_ineq, l_bounds])
    u_total = jnp.concatenate([u_eq, u_ineq, u_bounds])

    # Solve QP
    solver = OSQP()
    solution = solver.run(P, q, A_total, l_total, u_total)
    return solution.x

# %%
# ------------------------
# Loss and Utilities
# ------------------------
def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)

def soft_update(target_params: Any, online_params: Any, tau: float = 0.005) -> Any:
    return jax.tree_util.tree_map(
        lambda tp, op: tp * (1 - tau) + op * tau, target_params, online_params
    )

# %%
# ------------------------
# Action Retrieval Functions
# ------------------------
def get_id_actions(
    policy_id_params: Any,
    policy_id_model: nn.Module,
    s_id: jnp.ndarray,
    config: Config
) -> jnp.ndarray:
    # Build constraints
    A, b, Aeq, beq, lb, ub = h_.build_constraints_ID(
        s_id,
        config.Delta_ti,
        config.beta_pump,
        config.beta_turbine,
        config.c_pump_up,
        config.c_pump_down,
        config.c_turbine_up,
        config.c_turbine_down,
        config.x_min_pump,
        config.x_max_pump,
        config.x_min_turbine,
        config.x_max_turbine,
        config.Rmax,
    )
    # Apply policy with constraints
    actions = policy_id_model.apply(policy_id_params, s_id, Aeq, beq, A, b, ub, lb)
    return actions

def get_da_actions(
    policy_da_params: Any,
    policy_da_model: nn.Module,
    s_da: jnp.ndarray,
    config: Config
) -> jnp.ndarray:
    # Build constraints
    lb, ub = h_.build_constraints_DA(config.x_max_pump, config.x_max_turbine)
    # Apply policy with constraints
    actions = policy_da_model.apply(policy_da_params, s_da, ub, lb)
    return actions

# %%
# ------------------------
# Update Functions
# ------------------------
@jax.jit
def update_q_id(
    q_id_params: Any,
    q_id_opt_state: Any,
    q_id_target_params: Any,
    q_da_target_params: Any,
    policy_da_params: Any,
    q_id_model: nn.Module,
    q_da_model: nn.Module,
    s_id: jnp.ndarray,
    a_id: jnp.ndarray,
    r_id: jnp.ndarray,
    s_da_next: jnp.ndarray,
    config: Config,
    gamma: float = 0.99
) -> Tuple[Any, Any, jnp.ndarray]:
    # Compute Q_DA(s_{t+1}^{DA}, policy_DA(s_{t+1}^{DA}))
    next_da_actions = get_da_actions(policy_da_params, policy_da_model, s_da_next, config)
    q_da_values = q_da_model.apply(q_da_target_params, s_da_next, next_da_actions)
    q_target_id = r_id + gamma * q_da_values

    def loss_fn(params):
        q_estimate = q_id_model.apply(params, s_id, a_id)
        loss = mse_loss(q_estimate, q_target_id)
        return loss, q_estimate

    grads, (q_estimate) = jax.grad(loss_fn, has_aux=True)(q_id_params)
    updates, q_id_opt_state_new = q_id_opt.update(grads, q_id_opt_state)
    q_id_params_new = optax.apply_updates(q_id_params, updates)
    return q_id_params_new, q_id_opt_state_new, q_estimate

@jax.jit
def update_q_da(
    q_da_params: Any,
    q_da_opt_state: Any,
    q_da_target_params: Any,
    q_id_target_params: Any,
    policy_id_params: Any,
    q_da_model: nn.Module,
    q_id_model: nn.Module,
    s_da: jnp.ndarray,
    a_da: jnp.ndarray,
    r_da: jnp.ndarray,
    s_id_next: jnp.ndarray,
    config: Config,
    gamma: float = 0.99
) -> Tuple[Any, Any, jnp.ndarray]:
    # Compute Q_ID(s_{t}^{ID}, policy_ID(s_{t}^{ID}))
    next_id_actions = get_id_actions(policy_id_params, policy_id_model, s_id_next, config)
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
def update_policy_da(
    policy_da_params: Any,
    policy_da_opt_state: Any,
    q_da_params: Any,
    q_da_model: nn.Module,
    s_da: jnp.ndarray,
    config: Config
) -> Tuple[Any, Any]:
    # Deterministic policy gradient: maximize Q(s, pi(s))
    def loss_fn(params):
        a_da = policy_da_model.apply(
            params,
            s_da,
            jnp.ones((s_da.shape[0], config.length_R)) * config.x_max_pump,  # Adjust based on action_dim_da
            jnp.zeros((s_da.shape[0], config.length_R))                     # Adjust based on action_dim_da
        )
        q_values = q_da_model.apply(q_da_params, s_da, a_da)
        return -jnp.mean(q_values)

    grads = jax.grad(loss_fn)(policy_da_params)
    updates, policy_da_opt_state_new = policy_da_opt.update(grads, policy_da_opt_state)
    policy_da_params_new = optax.apply_updates(policy_da_params, updates)
    return policy_da_params_new, policy_da_opt_state_new

@jax.jit
def update_policy_id(
    policy_id_params: Any,
    policy_id_opt_state: Any,
    q_id_params: Any,
    q_id_model: nn.Module,
    s_id: jnp.ndarray,
    config: Config
) -> Tuple[Any, Any]:
    # Deterministic policy gradient: maximize Q(s, pi(s))
    def loss_fn(params):
        a_id = policy_id_model.apply(
            params,
            s_id,
            [],  # Aeq
            [],  # beq
            [],  # A
            [],  # b
            jnp.ones((s_id.shape[0], config.length_R)) * config.x_max_pump,  # Adjust based on action_dim_id
            jnp.zeros((s_id.shape[0], config.length_R))                     # Adjust based on action_dim_id
        )
        q_values = q_id_model.apply(q_id_params, s_id, a_id)
        return -jnp.mean(q_values)

    grads = jax.grad(loss_fn)(policy_id_params)
    updates, policy_id_opt_state_new = policy_id_opt.update(grads, policy_id_opt_state)
    policy_id_params_new = optax.apply_updates(policy_id_params, updates)
    return policy_id_params_new, policy_id_opt_state_new

# %%
# ------------------------
# Model and Optimizer Initialization
# ------------------------
# Initialize PRNGKeys
key = jax.random.PRNGKey(0)
da_key, id_key, pda_key, pid_key = jax.random.split(key, 4)

# Define state and action dimensions based on loaded data
# da_data = (states, actions, rewards, next_states)
states_da, actions_da, _, _ = da_data
states_id, actions_id, _, _ = id_data

state_dim_da = states_da.shape[1]  # Number of features in state
action_dim_da = actions_da.shape[1]  # Number of action dimensions

state_dim_id = states_id.shape[1]
action_dim_id = actions_id.shape[1]

# Initialize Models
q_da_model = QNetwork(state_dim=state_dim_da, action_dim=action_dim_da)
q_id_model = QNetwork(state_dim=state_dim_id, action_dim=action_dim_id)
policy_da_model = PolicyNetworkDA(state_dim=state_dim_da, action_dim=action_dim_da)
policy_id_model = PolicyNetworkID(state_dim=state_dim_id, action_dim=action_dim_id)

# Dummy inputs for initialization
dummy_s_da = jnp.ones((1, state_dim_da), dtype=jnp.float32)
dummy_a_da = jnp.ones((1, action_dim_da), dtype=jnp.float32)
dummy_s_id = jnp.ones((1, state_dim_id), dtype=jnp.float32)
dummy_a_id = jnp.ones((1, action_dim_id), dtype=jnp.float32)

# Initialize Parameters
q_da_params = q_da_model.init(da_key, dummy_s_da, dummy_a_da)
q_id_params = q_id_model.init(id_key, dummy_s_id, dummy_a_id)
policy_da_params = policy_da_model.init(
    pda_key,
    dummy_s_da,
    jnp.ones((1, action_dim_da)) * config.x_max_pump,
    jnp.zeros((1, action_dim_da))
)
policy_id_params = policy_id_model.init(
    pid_key,
    dummy_s_id,
    [], [], [], [],  # Empty constraints for initialization
    jnp.ones((1, action_dim_id)) * config.x_max_pump,
    jnp.zeros((1, action_dim_id))
)

# Initialize Target Networks
q_da_target_params = q_da_params
q_id_target_params = q_id_params

# Initialize Optimizers
q_da_opt = optax.adam(config.learning_rate)
q_id_opt = optax.adam(config.learning_rate)
policy_da_opt = optax.adam(config.learning_rate)
policy_id_opt = optax.adam(config.learning_rate)

q_da_opt_state = q_da_opt.init(q_da_params)
q_id_opt_state = q_id_opt.init(q_id_params)
policy_da_opt_state = policy_da_opt.init(policy_da_params)
policy_id_opt_state = policy_id_opt.init(policy_id_params)

# %%
# ------------------------
# Training Loop
# ------------------------
for epoch in range(config.num_epochs):
    # Train Q_ID and Policy_ID
    for s_id, a_id, r_id, s_da_next in batch_iter(id_data, config.batch_size):
        s_id = jnp.array(s_id, dtype=jnp.float32)
        a_id = jnp.array(a_id, dtype=jnp.float32)
        r_id = jnp.array(r_id, dtype=jnp.float32).reshape(-1, 1)
        s_da_next = jnp.array(s_da_next, dtype=jnp.float32)

        q_id_params, q_id_opt_state, _ = update_q_id(
            q_id_params,
            q_id_opt_state,
            q_id_target_params,
            q_da_target_params,
            policy_da_params,
            q_id_model,
            q_da_model,
            s_id,
            a_id,
            r_id,
            s_da_next,
            config,
            config.gamma
        )

        policy_id_params, policy_id_opt_state = update_policy_id(
            policy_id_params,
            policy_id_opt_state,
            q_id_params,
            q_id_model,
            s_id,
            config
        )

    # Train Q_DA and Policy_DA
    for s_da, a_da, r_da, s_id_next in batch_iter(da_data, config.batch_size):
        s_da = jnp.array(s_da, dtype=jnp.float32)
        a_da = jnp.array(a_da, dtype=jnp.float32)
        r_da = jnp.array(r_da, dtype=jnp.float32).reshape(-1, 1)
        s_id_next = jnp.array(s_id_next, dtype=jnp.float32)

        q_da_params, q_da_opt_state, _ = update_q_da(
            q_da_params,
            q_da_opt_state,
            q_da_target_params,
            q_id_target_params,
            policy_id_params,
            q_da_model,
            q_id_model,
            s_da,
            a_da,
            r_da,
            s_id_next,
            config,
            config.gamma
        )

        policy_da_params, policy_da_opt_state = update_policy_da(
            policy_da_params,
            policy_da_opt_state,
            q_da_params,
            q_da_model,
            s_da,
            config
        )

    # Soft update target networks
    q_da_target_params = soft_update(q_da_target_params, q_da_params, tau=0.005)
    q_id_target_params = soft_update(q_id_target_params, q_id_params, tau=0.005)

    print(f"Epoch {epoch + 1}/{config.num_epochs} finished.")

# %%
# ------------------------
# Action Sampling
# ------------------------
def sample_action_da(
    policy_da_params: Any,
    policy_da_model: nn.Module,
    s_da_example: jnp.ndarray,
    config: Config
) -> jnp.ndarray:
    return get_da_actions(policy_da_params, policy_da_model, s_da_example, config)

def sample_action_id(
    policy_id_params: Any,
    policy_id_model: nn.Module,
    s_id_example: jnp.ndarray,
    config: Config
) -> jnp.ndarray:
    return get_id_actions(policy_id_params, policy_id_model, s_id_example, config)

# %%
# ------------------------
# Example Usage
# ------------------------
s_da_example = jnp.ones((1, state_dim_da), dtype=jnp.float32)
da_action = sample_action_da(policy_da_params, policy_da_model, s_da_example, config)

s_id_example = jnp.ones((1, state_dim_id), dtype=jnp.float32)
id_action = sample_action_id(policy_id_params, policy_id_model, s_id_example, config)

print("Selected DA action (continuous):", da_action)
print("Selected ID action (continuous):", id_action)
