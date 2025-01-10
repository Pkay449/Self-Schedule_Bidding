# src/Sequential_NFQCA/utils/losses.py

import jax.numpy as jnp


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)
