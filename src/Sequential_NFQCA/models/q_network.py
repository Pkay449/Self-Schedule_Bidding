# src/Sequential_NFQCA/models/q_network.py

import jax.numpy as jnp
import flax.linen as nn

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