# %%
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

import os
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%

da_path = "Results/offline_dataset_day_ahead.pkl"
id_path = "Results/offline_dataset_intraday.pkl"

da_df = pkl.load(open(da_path, "rb"))
id_df = pkl.load(open(id_path, "rb"))

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
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxopt import OSQP
from flax import linen as nn
from functools import partial
import numpy as np
import pickle

import helper as h_

# ------------------------
# Parameters
# ------------------------
Season = "Summer"
length_R = 5
seed = 2
D = 7  # days in forecast
Rmax = 100
np.random.seed(seed)

t_ramp_pump_up = 2 / 60
t_ramp_pump_down = 2 / 60
t_ramp_turbine_up = 2 / 60
t_ramp_turbine_down = 2 / 60

c_grid_fee = 5 / 4
Delta_ti = 0.25
Delta_td = 1.0

Q_mult = 1.2
Q_fix = 3
Q_start_pump = 15
Q_start_turbine = 15

beta_pump = 0.9
beta_turbine = 0.9

x_max_pump = 10
x_min_pump = 5
x_max_turbine = 10
x_min_turbine = 5

R_vec = np.linspace(0, Rmax, length_R)
x_vec = np.array([-x_max_turbine, 0, x_max_pump])

c_pump_up = t_ramp_pump_up / 2
c_pump_down = t_ramp_pump_down / 2
c_turbine_up = t_ramp_turbine_up / 2
c_turbine_down = t_ramp_turbine_down / 2

# %%


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
    action_dim: int = 1177
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
    state_dim: int = 842
    action_dim: int = 24
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state, ub, lb):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        # Output a vector of action_dim floats (continuous actions)
        actions = nn.Dense(self.action_dim)(x)

        actions = lb + (ub - lb) * nn.sigmoid(actions)

        return actions


class PolicyID(nn.Module):
    state_dim: int = 890
    action_dim: int = 1177
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state, Aeq, beq, A, b, ub, lb):
        """
        Returns a vector of actions which satisfy the constraints

        Args:
            Aeq (List[np.ndarray]): List of matrices for equality constraints (Aeq_i).
            beq (List[np.ndarray]): List of vectors for equality constraints (beq_i).
            A (List[np.ndarray]): List of matrices for inequality constraints (A_i).
            b (List[np.ndarray]): List of vectors for inequality constraints (b_i).
            ub (np.ndarray): upper bound for actions (shape: (action_dim,))
            lb (np.ndarray): lower bound for actions (shape: (action_dim,))

        Returns:
            np.ndarray: actions returned by policy
        """
        # Produce raw unconstrained actions
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        raw_actions = nn.Dense(self.action_dim)(x)

        # Combine all equality constraints
        # Each Aeq_i @ actions = beq_i
        # Stack them vertically
        if len(Aeq) > 0:
            Aeq_all = jnp.vstack(Aeq)  # Shape: (sum_of_all_eq_rows, action_dim)
            beq_all = jnp.concatenate(beq)  # Shape: (sum_of_all_eq_rows,)
        else:
            # No equality constraints
            Aeq_all = jnp.zeros((0, self.action_dim))
            beq_all = jnp.zeros((0,))

        # Combine all inequality constraints
        # Each A_i @ actions <= b_i
        if len(A) > 0:
            A_all = jnp.vstack(A)  # Shape: (sum_of_all_ineq_rows, action_dim)
            b_all = jnp.concatenate(b)  # Shape: (sum_of_all_ineq_rows,)
        else:
            # No inequality constraints
            A_all = jnp.zeros((0, self.action_dim))
            b_all = jnp.zeros((0,))

        # Now we need to form a QP of the form:
        # minimize (1/2)*actions^T * I * actions - raw_actions^T * actions
        # s.t. Aeq_all * actions = beq_all
        #      A_all * actions <= b_all
        #      lb <= actions <= ub

        # Objective: Q = I, c = -raw_actions
        Q = jnp.eye(self.action_dim)
        c = -raw_actions

        # Convert equality constraints to OSQP form:
        # For equality: Aeq*x = beq can be represented as
        # l_eq = u_eq = beq
        A_eq_block = Aeq_all
        l_eq_block = beq_all
        u_eq_block = beq_all

        # For inequality: A*x <= b, we have
        # l_ineq = -∞, u_ineq = b
        A_ineq_block = A_all
        l_ineq_block = -jnp.inf * jnp.ones(A_all.shape[0])
        u_ineq_block = b_all

        # For bounds:
        # lb <= actions <= ub
        # This can be represented as:
        # I * actions <= ub   and   (-I)*actions <= -lb
        A_bounds_block = jnp.vstack(
            [jnp.eye(self.action_dim), -jnp.eye(self.action_dim)]
        )
        l_bounds_block = jnp.concatenate(
            [-jnp.inf * jnp.ones(self.action_dim), -jnp.inf * jnp.ones(self.action_dim)]
        )
        u_bounds_block = jnp.concatenate([ub, -lb])

        # Combine all constraints:
        A_all_blocks = jnp.vstack([A_eq_block, A_ineq_block, A_bounds_block])
        l_all_blocks = jnp.concatenate([l_eq_block, l_ineq_block, l_bounds_block])
        u_all_blocks = jnp.concatenate([u_eq_block, u_ineq_block, u_bounds_block])

        # Create an OSQP solver instance
        solver = OSQP()

        P = Q  # rename Q to P
        q = c  # rename c to q

        solution = solver.run(P, q, A_all_blocks, l_all_blocks, u_all_blocks)
        actions_feasible = solution.x



        return actions_feasible


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def soft_update(target_params, online_params, tau=0.005):
    return jax.tree_util.tree_map(
        lambda tp, op: tp * (1 - tau) + op * tau, target_params, online_params
    )


def get_id_actions(
    policy_id_params,
    s_id,
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
    # constraints
    A, b, Aeq, beq, lb, ub = h_.build_constraints_ID(
        s_id,
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
    # Apply the policy with constraints
    actions = policy_id_model.apply(policy_id_params, s_id, Aeq, beq, A, b, ub, lb)
    return actions


def get_da_actions(policy_da_params, s_da, x_max_pump, x_max_turbine):
    # constraints
    lb, ub = h_.build_constraints_DA(x_max_pump, x_max_turbine)
    # Apply the policy with constraints
    actions = policy_da_model.apply(policy_da_params, s_da, ub, lb)
    return actions


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
    # next_da_actions = policy_da_model.apply(policy_da_params, s_da_next)
    next_da_actions = get_da_actions(policy_da_params)
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
    # next_id_actions = policy_id_model.apply(policy_id_params, s_id_next)
    next_id_actions = get_id_actions(policy_id_params, s_id_next)
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


# =============================================================================

# Initialize PRNGKeys
key = jax.random.PRNGKey(0)
da_key, id_key, pda_key, pid_key = jax.random.split(key, 4)

# Dummy inputs for initialization (continuous actions)
dummy_s_da = jnp.ones((1, 842))
dummy_a_da = jnp.ones((1, 24), dtype=jnp.float32)
dummy_s_id = jnp.ones((1, 890))
dummy_a_id = jnp.ones((1, 1177), dtype=jnp.float32)

q_da_model = QNetworkDA()
q_id_model = QNetworkID()
policy_da_model = PolicyDA()
policy_id_model = PolicyID()

q_da_params = q_da_model.init(da_key, dummy_s_da, dummy_a_da)
q_id_params = q_id_model.init(id_key, dummy_s_id, dummy_a_id)
# For DA Policy:
dummy_ub_da = jnp.ones((24,), dtype=jnp.float32) * 10.0  # arbitrary upper bound
dummy_lb_da = jnp.zeros((24,), dtype=jnp.float32)        # arbitrary lower bound
policy_da_params = policy_da_model.init(pda_key, dummy_s_da, dummy_ub_da, dummy_lb_da)

# Dummy constraints for ID policy initialization
Aeq_dummy = []
beq_dummy = []
A_dummy = []
b_dummy = []
ub_dummy = jnp.ones((1177,), dtype=jnp.float32) * 10.0
lb_dummy = jnp.zeros((1177,), dtype=jnp.float32)

# Initialize PolicyID params with all required dummy arguments
policy_id_params = policy_id_model.init(pid_key, dummy_s_id, Aeq_dummy, beq_dummy, A_dummy, b_dummy, ub_dummy, lb_dummy)


q_da_target_params = q_da_params
q_id_target_params = q_id_params

learning_rate = 1e-3
q_da_opt = optax.adam(learning_rate)
q_id_opt = optax.adam(learning_rate)
policy_da_opt = optax.adam(learning_rate)
policy_id_opt = optax.adam(learning_rate)

q_da_opt_state = q_da_opt.init(q_da_params)
q_id_opt_state = q_id_opt.init(q_id_params)
policy_da_opt_state = policy_da_opt.init(policy_da_params)# For ID Policy:
# Let's say we have no constraints at initialization
Aeq_dummy = []
beq_dummy = []
A_dummy = []
b_dummy = []
ub_dummy = jnp.ones((1177,), dtype=jnp.float32) * 10.0
lb_dummy = jnp.zeros((1177,), dtype=jnp.float32)

policy_id_params = policy_id_model.init(pid_key, dummy_s_id, Aeq_dummy, beq_dummy, A_dummy, b_dummy, ub_dummy, lb_dummy)
policy_id_opt_state = policy_id_opt.init(policy_id_params)

gamma = 0.99
batch_size = 64
num_epochs = 10

da_data = load_offline_data_da("Results/offline_dataset_day_ahead.pkl")
id_data = load_offline_data_id("Results/offline_dataset_intraday.pkl")


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
    # actions = policy_da_model.apply(policy_da_params, s_da_example)
    actions = get_da_actions(policy_da_params)
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
