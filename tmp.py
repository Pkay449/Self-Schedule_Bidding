# main.py
import os
import warnings
import pickle
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

# ------------------------
# Configuration
# ------------------------
@dataclass
class Config:
    name: str
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    data_path: str = ""
    seed: int = 2

@dataclass
class TrainingParams:
    gamma: float = 0.99
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    tau: float = 0.005

# Load simulation parameters
sim_params = SimulationParams()

# Define configurations for Day-Ahead and Intraday
configs = {
    "DA": Config(
        name="DA",
        state_dim=842,
        action_dim=24,
        data_path="Results/offline_dataset_day_ahead.pkl",
        seed=sim_params.seed
    ),
    "ID": Config(
        name="ID",
        state_dim=890,
        action_dim=1152,
        data_path="Results/offline_dataset_intraday.pkl",
        seed=sim_params.seed
    ),
}

training_params = TrainingParams()

# Set random seed
np.random.seed(configs["DA"].seed)

# ------------------------
# Data Loading
# ------------------------
def load_offline_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        df = pickle.load(f)
    states = np.stack(df["state"].values)
    actions = np.stack(df["action"].values).astype(np.float32)
    rewards = df["reward"].values.astype(np.float32)
    next_states = np.stack(df["next_state"].values)
    return states, actions, rewards, next_states

def batch_iter(data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], batch_size: int, shuffle: bool = True):
    states, actions, rewards, next_states = data
    N = len(states)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        idx = indices[start:start + batch_size]
        yield (states[idx], actions[idx], rewards[idx], next_states[idx])

# Load data for both DA and ID
data = {key: load_offline_data(cfg.data_path) for key, cfg in configs.items()}

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

class PolicyNetwork(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

# ------------------------
# Initialization
# ------------------------
def initialize_models(configs: Dict[str, Config], key: jax.random.PRNGKey) -> Dict[str, Dict[str, Any]]:
    models = {}
    keys = jax.random.split(key, len(configs)*2)  # Q and Policy for each config
    for i, (name, cfg) in enumerate(configs.items()):
        q_model = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim)
        policy_model = PolicyNetwork(state_dim=cfg.state_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim)
        
        q_key, policy_key = jax.random.split(keys[i*2])
        policy_key, _ = jax.random.split(policy_key)
        
        dummy_state = jnp.ones((1, cfg.state_dim))
        dummy_action = jnp.ones((1, cfg.action_dim))
        
        q_params = q_model.init(q_key, dummy_state, dummy_action)
        policy_params = policy_model.init(policy_key, dummy_state)
        
        models[name] = {
            "Q": q_model,
            "Policy": policy_model,
            "Q_params": q_params,
            "Policy_params": policy_params,
            "Q_target_params": q_params,
            "Policy_target_params": policy_params,
        }
    return models

# Initialize PRNGKeys
key = jax.random.PRNGKey(0)
models = initialize_models(configs, key)

# Initialize optimizers
optimizers = {}
for name in configs.keys():
    optimizers[name] = {
        "Q_optimizer": optax.adam(training_params.learning_rate),
        "Policy_optimizer": optax.adam(training_params.learning_rate),
    }
    optimizers[name]["Q_opt_state"] = optimizers[name]["Q_optimizer"].init(models[name]["Q_params"])
    optimizers[name]["Policy_opt_state"] = optimizers[name]["Policy_optimizer"].init(models[name]["Policy_params"])

# ------------------------
# Utility Functions
# ------------------------
def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)

def soft_update(target_params: Dict, online_params: Dict, tau: float = 0.005) -> Dict:
    return jax.tree_util.tree_map(lambda tp, op: tp * (1 - tau) + op * tau, target_params, online_params)

# ------------------------
# Update Functions
# ------------------------
@jax.jit
def update_q(
    q_model: QNetwork,
    q_params: Dict,
    q_opt: optax.GradientTransformation,
    q_opt_state: optax.OptState,
    q_target_params: Dict,
    policy_model: PolicyNetwork,
    policy_params: Dict,
    target_q_model: QNetwork,
    target_q_params: Dict,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    s_next: jnp.ndarray,
    gamma: float
) -> Tuple[Dict, optax.OptState, jnp.ndarray]:
    # Compute target actions and Q-values
    next_a = policy_model.apply(policy_params, s_next)
    q_target = target_q_model.apply(target_q_params, s_next, next_a)
    target = r + gamma * q_target.squeeze(-1)
    
    def loss_fn(params):
        q_estimate = q_model.apply(params, s, a).squeeze(-1)
        loss = mse_loss(q_estimate, target)
        return loss, q_estimate
    
    grads, (loss, _) = jax.grad(loss_fn, has_aux=True)(q_params)
    updates, new_opt_state = q_opt.update(grads, q_opt_state)
    new_q_params = optax.apply_updates(q_params, updates)
    return new_q_params, new_opt_state, loss

@jax.jit
def update_policy(
    policy_model: PolicyNetwork,
    policy_params: Dict,
    policy_opt: optax.GradientTransformation,
    policy_opt_state: optax.OptState,
    q_model: QNetwork,
    q_params: Dict,
    s: jnp.ndarray
) -> Tuple[Dict, optax.OptState, jnp.ndarray]:
    def loss_fn(params):
        a = policy_model.apply(params, s)
        q_val = q_model.apply(q_params, s, a).squeeze(-1)
        loss = -jnp.mean(q_val)
        return loss, q_val

    grads, (loss, _) = jax.grad(loss_fn, has_aux=True)(policy_params)
    updates, new_opt_state = policy_opt.update(grads, policy_opt_state)
    new_policy_params = optax.apply_updates(policy_params, updates)
    return new_policy_params, new_opt_state, loss

# ------------------------
# Training Loop
# ------------------------
def train(models: Dict[str, Dict[str, Any]], optimizers: Dict[str, Dict[str, Any]], data: Dict[str, Tuple], training_params: TrainingParams, configs: Dict[str, Config]):
    for epoch in range(training_params.num_epochs):
        for name, cfg in configs.items():
            q_model = models[name]["Q"]
            policy_model = models[name]["Policy"]
            q_params = models[name]["Q_params"]
            policy_params = models[name]["Policy_params"]
            q_target_params = models[name]["Q_target_params"]
            policy_target_params = models[name]["Policy_target_params"]
            
            q_opt = optimizers[name]["Q_optimizer"]
            q_opt_state = optimizers[name]["Q_opt_state"]
            policy_opt = optimizers[name]["Policy_optimizer"]
            policy_opt_state = optimizers[name]["Policy_opt_state"]

            # Determine the counterpart for target Q (DA <-> ID)
            counterpart = "ID" if name == "DA" else "DA"
            target_q_model = models[counterpart]["Q"]
            target_q_params = models[counterpart]["Q_target_params"]
            # Note: In the original code, policy_target_params aren't used in Q update
            # You may need to decide whether to use target policy or current policy for the target actions
            target_policy_model = models[counterpart]["Policy"]
            target_policy_params = models[counterpart]["Policy_params"]

            # Training Q-network
            for batch in batch_iter(data[name], training_params.batch_size):
                s, a, r, s_next = batch
                s = jnp.array(s, dtype=jnp.float32)
                a = jnp.array(a, dtype=jnp.float32)
                r = jnp.array(r, dtype=jnp.float32).reshape(-1, 1)
                s_next = jnp.array(s_next, dtype=jnp.float32)

                q_params, q_opt_state, loss_q = update_q(
                    q_model,
                    q_params,
                    q_opt,
                    q_opt_state,
                    q_target_params,
                    policy_model,
                    policy_params,
                    target_q_model,
                    target_q_params,
                    s,
                    a,
                    r,
                    s_next,
                    training_params.gamma
                )

            # Update optimizers
            optimizers[name]["Q_params"] = q_params
            optimizers[name]["Q_opt_state"] = q_opt_state

            # Training Policy
            for batch in batch_iter(data[name], training_params.batch_size):
                s, _, _, _ = batch
                s = jnp.array(s, dtype=jnp.float32)

                policy_params, policy_opt_state, loss_policy = update_policy(
                    policy_model,
                    policy_params,
                    policy_opt,
                    policy_opt_state,
                    q_model,
                    q_params,
                    s
                )

            # Update optimizers
            optimizers[name]["Policy_params"] = policy_params
            optimizers[name]["Policy_opt_state"] = policy_opt_state

            # Soft update target networks
            models[name]["Q_target_params"] = soft_update(models[name]["Q_target_params"], q_params, training_params.tau)
            models[name]["Policy_target_params"] = soft_update(models[name]["Policy_target_params"], policy_params, training_params.tau)

        print(f"Epoch {epoch + 1}/{training_params.num_epochs} completed.")

# Execute training
train(models, optimizers, data, training_params, configs)

# ------------------------
# Policy Usage
# ------------------------
def sample_action(policy_model: PolicyNetwork, policy_params: Dict, s_example: jnp.ndarray) -> jnp.ndarray:
    return policy_model.apply(policy_params, s_example)

# Sample actions for DA and ID
s_da_example = jnp.ones((1, configs["DA"].state_dim), dtype=jnp.float32)
da_action = sample_action(models["DA"]["Policy"], models["DA"]["Policy_params"], s_da_example)

s_id_example = jnp.ones((1, configs["ID"].state_dim), dtype=jnp.float32)
id_action = sample_action(models["ID"]["Policy"], models["ID"]["Policy_params"], s_id_example)

print("Selected DA action (continuous):", da_action)
print("Selected ID action (continuous):", id_action)



    lb = np.concatenate(
        [
            np.zeros(96),
            -np.inf * np.ones(96),
            np.zeros(96 * 10)
        ]
    )
    
    ub = np.concatenate(
        [
            Rmax * np.ones(96),
            np.inf * np.ones(96 * 7),
            np.ones(96 * 4),
        ]
    )