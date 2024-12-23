# train.py
# %%
import os
import warnings
import pickle as pkl
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as np
from functools import partial

from config import SimulationParams
from helper import build_constraints_batch
from eval_learned_policy import eval_learned_policy

# %%

import jax
import jax.numpy as jnp
from jaxopt import OSQP


def qp_projection(raw_actions, A, b, Aeq, beq, lb, ub, solver, relaxation=10):
    """
    Solve the QP:
        minimize (1/2) * ||x - raw_actions||^2
        subject to A x <= b,
                   Aeq x = beq,
                   lb <= x <= ub.
    """
    # [Same as before, but use the passed solver]
    dim = raw_actions.shape[-1]
    P = jnp.eye(dim)
    q = -raw_actions

    # We have the problem in the form:
    # minimize 0.5 x^T P x + q^T x
    # subject to A x <= b,
    #            Aeq x = beq,
    #            lb <= x <= ub.

    # A (num constraints x dim), b (num constraints)
    # [A, I, -I] x <= [b, ub, -lb]
    I = jnp.eye(dim)
    # A = jnp.vstack([A, I, -I])
    # b = jnp.concatenate([b, ub, -lb])

    # A = b
    # |A - b| <= relaxation
    # A - b <= relaxation and A - b >= - relaxation
    # A <= b + rel and A >= b - rel
    # A <= b + rel and -A <= -b + rel

    A = jnp.vstack([A, I, -I, Aeq, -Aeq])
    b = jnp.concatenate([b, ub, -lb, beq + relaxation, -beq + relaxation])

    # Solve the QP using the existing solver
    solution = solver.run(
        params_obj=(P, q),
        # params_eq=(Aeq, beq),
        params_ineq=(A, b),
    )

    x_proj = solution.params.primal
    return x_proj


# %%

# Suppress warnings and set working directory
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------
# Load Offline Data
# ----------------------------------------------------
def load_offline_data(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads offline dataset from a pickle file. The file is expected to contain a dictionary
    with keys: "state", "action", "reward", "next_state". Each value should be a Series
    or array-like from which we can extract arrays.

    Returns:
        states, actions, rewards, next_states
    """
    with open(path, "rb") as f:
        df = pkl.load(f)
    states = np.stack(df["state"].values)
    actions = np.stack(df["action"].values)
    rewards = df["reward"].values
    next_states = np.stack(df["next_state"].values)
    return states, actions, rewards, next_states


def batch_iter(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
):
    """
    Generator that yields mini-batches of data.
    """
    states, actions, rewards, next_states = data
    N = len(states)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        idx = indices[start : start + batch_size]
        yield (states[idx], actions[idx], rewards[idx], next_states[idx])


# ----------------------------------------------------
# Model Definitions
# ----------------------------------------------------
class QNetwork(nn.Module):
    """
    A simple Q-network that takes continuous states and actions as inputs
    and outputs a scalar Q-value.
    """

    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action.astype(jnp.float32)], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class PolicyDA(nn.Module):
    """
    Policy network for the Day-Ahead scenario with bounded continuous actions.
    Actions are scaled via a sigmoid to fit within [lb, ub].
    """

    ub: float
    lb: float
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        actions = nn.Dense(self.action_dim)(x)
        # Scale to [lb, ub]
        actions = self.lb + (self.ub - self.lb) * nn.sigmoid(actions)
        return actions


class PolicyID(nn.Module):
    """
    Policy network for the Intraday scenario with QP-based projection to enforce constraints.
    """

    sim_params: SimulationParams  # Include simulation parameters
    lb: jnp.ndarray
    ub: jnp.ndarray
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    def setup(self):
        # Precompute masks as numpy boolean arrays (static)
        self.bounded_mask_np = ((self.lb > NEG_INF) & (self.ub < POS_INF))
        self.lower_bounded_mask_np = ((self.lb > NEG_INF) & (self.ub == POS_INF))
        self.upper_bounded_mask_np = ((self.lb == NEG_INF) & (self.ub < POS_INF))
        self.unbounded_mask_np = ((self.lb == NEG_INF) & (self.ub == POS_INF))

        # Convert to JAX boolean arrays
        self.bounded_mask = jnp.array(self.bounded_mask_np, dtype=bool)
        self.lower_bounded_mask = jnp.array(self.lower_bounded_mask_np, dtype=bool)
        self.upper_bounded_mask = jnp.array(self.upper_bounded_mask_np, dtype=bool)
        self.unbounded_mask = jnp.array(self.unbounded_mask_np, dtype=bool)

        # Verify that masks cover all action dimensions
        # total_masks = jnp.sum(self.bounded_mask) + jnp.sum(self.lower_bounded_mask) + \
        #               jnp.sum(self.upper_bounded_mask) + jnp.sum(self.unbounded_mask)
        # assert total_masks == self.action_dim, \
        #     f"Sum of all masks ({total_masks}) does not equal action_dim ({self.action_dim})"


    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the PolicyID network with efficient action bounding.

        Args:
            state: Input state, shape (batch_size, state_dim)

        Returns:
            Projected actions, shape (batch_size, action_dim)
        """
        # 1. Neural Network to generate raw actions
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        raw_actions = nn.Dense(self.action_dim)(x)  # Shape: (batch_size, action_dim)

        # 2. Apply transformations based on mask types using jnp.where

        # a. Fully Bounded: [lb, ub]
        bounded_actions = self.lb + (self.ub - self.lb) * nn.sigmoid(raw_actions)
        actions = jnp.where(self.bounded_mask, bounded_actions, raw_actions)

        # b. Lower Bounded Only: [lb, +inf)
        lower_bounded_actions = self.lb + nn.softplus(raw_actions)
        actions = jnp.where(self.lower_bounded_mask, lower_bounded_actions, actions)

        # c. Upper Bounded Only: (-inf, ub]
        upper_bounded_actions = self.ub - nn.softplus(raw_actions)
        actions = jnp.where(self.upper_bounded_mask, upper_bounded_actions, actions)

        # d. Unbounded: (-inf, +inf) - already set to raw_actions

        return actions


# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------
def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


def soft_update(target_params, online_params, tau=0.005):
    return jax.tree_util.tree_map(
        lambda tp, op: tp * (1 - tau) + op * tau, target_params, online_params
    )


# ----------------------------------------------------
# Main Training Logic
# ----------------------------------------------------
sim_params = SimulationParams()

# Load datasets
da_data = load_offline_data("Results/offline_dataset_day_ahead.pkl")
id_data = load_offline_data("Results/offline_dataset_intraday.pkl")

# Bounds for PolicyID
NEG_INF = -1e8
POS_INF = 1e8

lb_id = np.concatenate(
    [
        np.zeros(96),  # bounded
        NEG_INF * np.ones(96),  # unbounded (lower bound ~ -inf)
        np.zeros(96 * 10),  # bounded
    ]
).astype(np.float32)

ub_id = np.concatenate(
    [
        sim_params.Rmax * np.ones(96),  # bounded
        POS_INF * np.ones(96 * 7),  # unbounded (upper bound ~ inf)
        np.ones(96 * 4),  # bounded
    ]
).astype(np.float32)

# After defining lb_id and ub_id
total_masks = (
    ((lb_id > NEG_INF) & (ub_id < POS_INF)).sum() +
    ((lb_id > NEG_INF) & (ub_id == POS_INF)).sum() +
    ((lb_id == NEG_INF) & (ub_id < POS_INF)).sum() +
    ((lb_id == NEG_INF) & (ub_id == POS_INF)).sum()
)

assert total_masks == lb_id.shape[0], \
    f"Sum of all masks ({total_masks}) does not equal action_dim ({lb_id.shape[0]})"


# Initialize models
key = jax.random.PRNGKey(0)
da_key, id_key, pda_key, pid_key = jax.random.split(key, 4)

q_da_model = QNetwork(state_dim=842, action_dim=24)
q_id_model = QNetwork(state_dim=890, action_dim=1152)
policy_da_model = PolicyDA(
    ub=sim_params.x_max_pump, lb=-sim_params.x_max_turbine, state_dim=842, action_dim=24
)
policy_id_model = PolicyID(
    sim_params=sim_params,
    lb=jnp.array(lb_id),
    ub=jnp.array(ub_id),
    state_dim=890,
    action_dim=1152,
)

dummy_s_da = jnp.ones((1, 842))
dummy_a_da = jnp.ones((1, 24), dtype=jnp.float32)
dummy_s_id = jnp.ones((1, 890))
dummy_a_id = jnp.ones((1, 1152), dtype=jnp.float32)

q_da_params = q_da_model.init(da_key, dummy_s_da, dummy_a_da)
q_id_params = q_id_model.init(id_key, dummy_s_id, dummy_a_id)
policy_da_params = policy_da_model.init(pda_key, dummy_s_da)
policy_id_params = policy_id_model.init(pid_key, dummy_s_id)

q_da_target_params = q_da_params
q_id_target_params = q_id_params

# Optimizers
q_learning_rate = 1e-4
policy_learning_rate = 1e-5

q_da_opt = optax.adam(q_learning_rate)
q_id_opt = optax.adam(q_learning_rate)
policy_da_opt = optax.adam(policy_learning_rate)
policy_id_opt = optax.adam(policy_learning_rate)

q_da_opt_state = q_da_opt.init(q_da_params)
q_id_opt_state = q_id_opt.init(q_id_params)
policy_da_opt_state = policy_da_opt.init(policy_da_params)
policy_id_opt_state = policy_id_opt.init(policy_id_params)

gamma = 0.99
batch_size = 256
num_epochs = 50


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
    """
    Update Q_ID by fitting to the Bellman target:
    Q_ID(s,a) -> r_ID + gamma * Q_DA(s', policy_DA(s'))
    """
    next_da_actions = policy_da_model.apply(policy_da_params, s_da_next)
    q_da_values = q_da_model.apply(q_da_target_params, s_da_next, next_da_actions)
    q_target_id = r_id + gamma * q_da_values

    def loss_fn(params):
        q_estimate = q_id_model.apply(params, s_id, a_id)
        return mse_loss(q_estimate, q_target_id), q_estimate

    grads, q_estimate = jax.grad(loss_fn, has_aux=True)(q_id_params)
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
    """
    Update Q_DA by fitting to the Bellman target:
    Q_DA(s,a) -> r_DA + gamma * Q_ID(s', policy_ID(s'))
    """
    next_id_actions = policy_id_model.apply(policy_id_params, s_id_next)
    q_id_values = q_id_model.apply(q_id_target_params, s_id_next, next_id_actions)
    q_target_da = r_da + gamma * q_id_values

    def loss_fn(params):
        q_da_values = q_da_model.apply(params, s_da, a_da)
        return mse_loss(q_da_values, q_target_da), q_da_values

    grads, q_da_values = jax.grad(loss_fn, has_aux=True)(q_da_params)
    updates, q_da_opt_state_new = q_da_opt.update(grads, q_da_opt_state)
    q_da_params_new = optax.apply_updates(q_da_params, updates)
    return q_da_params_new, q_da_opt_state_new, q_da_values


@jax.jit
def update_policy_da(policy_da_params, policy_da_opt_state, q_da_params, s_da):
    """
    Update Policy_DA by maximizing Q_DA(s, policy_DA(s)).
    """

    def loss_fn(params):
        a_da = policy_da_model.apply(params, s_da)
        q_values = q_da_model.apply(q_da_params, s_da, a_da)
        return -jnp.mean(q_values)

    grads = jax.grad(loss_fn)(policy_da_params)
    updates, policy_da_opt_state_new = policy_da_opt.update(grads, policy_da_opt_state)
    policy_da_params_new = optax.apply_updates(policy_da_params, updates)
    return policy_da_params_new, policy_da_opt_state_new


@jax.jit
def update_policy_id(policy_id_params, policy_id_opt_state, q_id_params, s_id):
    """
    Update Policy_ID by maximizing Q_ID(s, policy_ID(s)).
    """

    def loss_fn(params):
        a_id = policy_id_model.apply(params, s_id)
        q_values = q_id_model.apply(q_id_params, s_id, a_id)
        return -jnp.mean(q_values)

    grads = jax.grad(loss_fn)(policy_id_params)
    updates, policy_id_opt_state_new = policy_id_opt.update(grads, policy_id_opt_state)
    policy_id_params_new = optax.apply_updates(policy_id_params, updates)
    return policy_id_params_new, policy_id_opt_state_new


from functools import partial


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
def update_policy_id_with_penalty(
    policy_id_params,
    policy_id_opt_state,
    q_id_params,
    states_id,
    Delta_ti,
    beta_pump,
    beta_turbine,
    c_pump_up,
    c_pump_down,
    c_turbine_up,
    c_turbine_down,
    x_min_pump,
    x_max_pump,
    x_min_turbine,
    x_max_turbine,
    Rmax,
):
    """
    Update Policy_ID by maximizing Q_ID(s, policy_ID(s)) with penalty for constraint violations.
    """

    def loss_fn(params):
        a_id = policy_id_model.apply(params, states_id)
        q_values = q_id_model.apply(q_id_params, states_id, a_id)

        # Compute constraints
        A, b, Aeq, beq, lb, ub = build_constraints_batch(
            states_id,
            Delta_ti,
            beta_pump,
            beta_turbine,
            c_pump_up,
            c_pump_down,
            c_turbine_up,
            c_turbine_down,
            x_min_pump,
            x_max_pump,
            x_min_turbine,
            x_max_turbine,
            Rmax,
        )
        batch_size = A.shape[0]
        action_size = a_id.shape[-1]
        relaxation = 1e2

        # Create identity matrix with batch dimension
        I = jnp.eye(action_size)
        I = jnp.expand_dims(I, axis=0)  # Shape: (1, action_size, action_size)
        I = jnp.tile(I, (batch_size, 1, 1))  # Shape: (batch_size, action_size, action_size)

        # # Concatenate all constraints
        A = jnp.concatenate([A, Aeq, -Aeq], axis=1)
        b = jnp.concatenate([b, beq + relaxation, -beq + relaxation], axis=1)


        # Penalty for A * x <= b
        Ax = jnp.einsum("bkc,bc->bk", A, a_id)  # Shape: (batch_size, num_constraints)
        penalty_ineq = jnp.maximum(Ax - b, 0.0)
        # penalty_eq = jnp.abs(jnp.einsum("bkc,bc->bk", Aeq, a_id) - beq)
        # penalty_ub = jnp.maximum(a_id - ub, 0.0)
        # penalty_lb = jnp.maximum(lb - a_id, 0.0)

        # Aggregate penalties
        penalty = jnp.sum(penalty_ineq**2)
        # penalty = jnp.sum(jnp.abs(penalty_ineq))
        # penalty = jnp.sum(-jnp.log(b - Ax + 1e-8))



        # Total loss
        return -jnp.mean(q_values) + penalty


    grads = jax.grad(loss_fn)(policy_id_params)
    updates, policy_id_opt_state_new = policy_id_opt.update(grads, policy_id_opt_state)
    policy_id_params_new = optax.apply_updates(policy_id_params, updates)
    return policy_id_params_new, policy_id_opt_state_new


# ----------------------------------------------------
# Training Loop
# ----------------------------------------------------
for epoch in range(num_epochs):
    # Train Q_ID and Policy_ID
    for s_id, a_id, r_id, s_da_next in batch_iter(id_data, batch_size):
        s_id = jnp.array(s_id, dtype=jnp.float32)
        a_id = jnp.array(a_id, dtype=jnp.float32)
        r_id = jnp.array(r_id, dtype=jnp.float32).reshape(-1, 1)
        s_da_next = jnp.array(s_da_next, dtype=jnp.float32)

        # Update Q_ID
        for i in range(5):
            q_id_params, q_id_opt_state, _ = update_q_id(
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

        # Update Policy_ID with penalties
        policy_id_params, policy_id_opt_state = update_policy_id_with_penalty(
            policy_id_params,
            policy_id_opt_state,
            q_id_params,
            s_id,
            sim_params.Delta_ti,
            sim_params.beta_pump,
            sim_params.beta_turbine,
            sim_params.c_pump_up,
            sim_params.c_pump_down,
            sim_params.c_turbine_up,
            sim_params.c_turbine_down,
            sim_params.x_min_pump,
            sim_params.x_max_pump,
            sim_params.x_min_turbine,
            sim_params.x_max_turbine,
            sim_params.Rmax,
        )

    # Train Q_DA and Policy_DA
    for s_da, a_da, r_da, s_id_next in batch_iter(da_data, batch_size):
        s_da = jnp.array(s_da, dtype=jnp.float32)
        a_da = jnp.array(a_da, dtype=jnp.float32)
        r_da = jnp.array(r_da, dtype=jnp.float32).reshape(-1, 1)
        s_id_next = jnp.array(s_id_next, dtype=jnp.float32)

        # Update Q_DA
        for i in range(5):
            q_da_params, q_da_opt_state, _ = update_q_da(
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

        # Update Policy_DA
        policy_da_params, policy_da_opt_state = update_policy_da(
            policy_da_params, policy_da_opt_state, q_da_params, s_da
        )

    # Soft updates of target networks
    q_da_target_params = soft_update(q_da_target_params, q_da_params)
    q_id_target_params = soft_update(q_id_target_params, q_id_params)

    print(f"Epoch {epoch+1}/{num_epochs} completed.")


# %%
# ----------------------------------------------------
# Action Sampling (Inference)
# ----------------------------------------------------
def sample_action_da(params, s_da_example: jnp.ndarray) -> jnp.ndarray:
    return policy_da_model.apply(params, s_da_example)


def sample_action_id(policy_id_params, states_id_example, config):
    raw_actions = policy_id_model.apply(policy_id_params, states_id_example)
    # Optionally, apply projection or clipping if needed
    # For penalty methods, this is usually not required
    return raw_actions


# Example inference
s_da_example = jnp.ones((1, 842), dtype=jnp.float32)
da_action = sample_action_da(policy_da_params, s_da_example)

s_id_example = jnp.ones((1, 890), dtype=jnp.float32)
id_action = sample_action_id(policy_id_params, s_id_example, sim_params)

print("Sample DA action:", da_action)
print("Sample ID action:", id_action)


# Example inference
s_da_example = jnp.ones((1, 842), dtype=jnp.float32)
da_action = sample_action_da(policy_da_params, s_da_example)

s_id_example = jnp.ones((1, 890), dtype=jnp.float32)
id_action = sample_action_id(policy_id_params, s_id_example, sim_params)

print("Sample DA action:", da_action)
print("Sample ID action:", id_action)
eval_learned_policy(
    policy_id_model, policy_da_model, policy_id_params, policy_da_params
)

# plot paths
import matplotlib.pyplot as plt

R_path = np.load("Results/BACKTEST_R_path.npy").ravel()
x_intraday_path = np.load("Results/BACKTEST_x_intraday_path.npy").ravel()
P_day_path = np.load("Results/BACKTEST_P_day_path.npy").ravel()
P_intraday_path = np.load("Results/BACKTEST_P_intraday_path.npy").ravel()
x_pump_path = np.load("Results/BACKTEST_x_pump_path.npy").ravel()
x_turbine_path = np.load("Results/BACKTEST_x_turbine_path.npy").ravel()
y_pump_path = np.load("Results/BACKTEST_y_pump_path.npy").ravel()
y_turbine_path = np.load("Results/BACKTEST_y_turbine_path.npy").ravel()
z_pump_path = np.load("Results/BACKTEST_z_pump_path.npy").ravel()

# Create a figure with multiple subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("Backtest Data Plots", fontsize=16)

# Plot each array in a subplot
data = {
    "R Path": R_path,
    "x Intraday Path": x_intraday_path,
    "P Day Path": P_day_path,
    "P Intraday Path": P_intraday_path,
    "x Pump Path": x_pump_path,
    "x Turbine Path": x_turbine_path,
    "y Pump Path": y_pump_path,
    "y Turbine Path": y_turbine_path,
    "z Pump Path": z_pump_path,
}

# Iterate through data and subplots
for ax, (title, array) in zip(axs.flat, data.items()):
    ax.plot(array)
    ax.set_title(title)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Values")
    ax.grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Results/backtest_plots.png")



# %%
