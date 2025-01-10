# src/Sequential_NFQCA/models/policy_id.py

import flax.linen as nn
import jax.numpy as jnp

from src.config import SimulationParams
from src.config import TrainingParams

training_params = TrainingParams()
NEG_INF = training_params.NEG_INF
POS_INF = training_params.POS_INF


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
        self.bounded_mask_np = (self.lb > NEG_INF) & (self.ub < POS_INF)
        self.lower_bounded_mask_np = (self.lb > NEG_INF) & (self.ub == POS_INF)
        self.upper_bounded_mask_np = (self.lb == NEG_INF) & (self.ub < POS_INF)
        self.unbounded_mask_np = (self.lb == NEG_INF) & (self.ub == POS_INF)

        # Convert to JAX boolean arrays
        self.bounded_mask = jnp.array(self.bounded_mask_np, dtype=bool)
        self.lower_bounded_mask = jnp.array(self.lower_bounded_mask_np, dtype=bool)
        self.upper_bounded_mask = jnp.array(self.upper_bounded_mask_np, dtype=bool)
        self.unbounded_mask = jnp.array(self.unbounded_mask_np, dtype=bool)

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
