# src/Sequential_NFQCA/utils/helper_functions.py

import jax
import jax.numpy as jnp

def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)

def soft_update(target_params, online_params, tau=0.005):
    return jax.tree_util.tree_map(
        lambda tp, op: tp * (1 - tau) + op * tau, target_params, online_params
    )
