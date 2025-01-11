# src/Sequential_NFQCA/training/trainer.py

import jax
import jax.numpy as jnp
import optax

from src.config import SimulationParams, TrainingParams
from src.Sequential_NFQCA.constraints.constraints import build_constraints_batch
from src.Sequential_NFQCA.models.policy_da import PolicyDA
from src.Sequential_NFQCA.models.policy_id import PolicyID
from src.Sequential_NFQCA.models.q_network import QNetwork
from src.Sequential_NFQCA.training.optimizers import get_optimizer
from src.Sequential_NFQCA.utils.data_loader import batch_iter
from src.Sequential_NFQCA.utils.helper_functions import mse_loss, soft_update


class NFQCA:
    def __init__(
        self, sim_params: SimulationParams, training_params: TrainingParams, rng_key
    ):
        self.sim_params = sim_params
        self.training_params = training_params

        # Initialize models
        self.q_da_model = QNetwork(state_dim=842, action_dim=24)
        self.q_id_model = QNetwork(state_dim=890, action_dim=1152)

        self.policy_da_model = PolicyDA(
            ub=sim_params.x_max_pump,
            lb=-sim_params.x_max_turbine,
            state_dim=842,
            action_dim=24,
        )
        self.policy_id_model = PolicyID(
            sim_params=sim_params,
            lb=jnp.array(self.get_lb_id()),
            ub=jnp.array(self.get_ub_id()),
            state_dim=890,
            action_dim=1152,
        )

        # Initialize parameters
        dummy_s_da = jnp.ones((1, 842))
        dummy_a_da = jnp.ones((1, 24), dtype=jnp.float32)
        dummy_s_id = jnp.ones((1, 890))
        dummy_a_id = jnp.ones((1, 1152), dtype=jnp.float32)

        keys = jax.random.split(rng_key, 4)
        self.q_da_params = self.q_da_model.init(keys[0], dummy_s_da, dummy_a_da)
        self.q_id_params = self.q_id_model.init(keys[1], dummy_s_id, dummy_a_id)
        self.policy_da_params = self.policy_da_model.init(keys[2], dummy_s_da)
        self.policy_id_params = self.policy_id_model.init(keys[3], dummy_s_id)

        # Initialize target networks
        self.q_da_target_params = self.q_da_params
        self.q_id_target_params = self.q_id_params

        # Initialize optimizers using TrainingParams
        self.q_da_opt = get_optimizer(self.training_params.q_learning_rate)
        self.q_id_opt = get_optimizer(self.training_params.q_learning_rate)
        self.policy_da_opt = get_optimizer(self.training_params.policy_learning_rate)
        self.policy_id_opt = get_optimizer(self.training_params.policy_learning_rate)

        self.q_da_opt_state = self.q_da_opt.init(self.q_da_params)
        self.q_id_opt_state = self.q_id_opt.init(self.q_id_params)
        self.policy_da_opt_state = self.policy_da_opt.init(self.policy_da_params)
        self.policy_id_opt_state = self.policy_id_opt.init(self.policy_id_params)

    def get_lb_id(self):
        """
        Construct the lower-bound array for the intraday policy.
        We now pull NEG_INF from self.training_params for clarity.
        """
        NEG_INF = self.training_params.NEG_INF
        lb_id = jnp.concatenate(
            [
                jnp.zeros(96),  # bounded
                NEG_INF * jnp.ones(96),  # unbounded
                jnp.zeros(96 * 10),  # bounded
            ]
        )
        return lb_id

    def get_ub_id(self):
        """
        Construct the upper-bound array for the intraday policy.
        We now pull POS_INF from self.training_params.
        """
        POS_INF = self.training_params.POS_INF
        ub_id = jnp.concatenate(
            [
                self.sim_params.Rmax * jnp.ones(96),  # bounded
                POS_INF * jnp.ones(96 * 7),  # unbounded
                jnp.ones(96 * 4),  # bounded
            ]
        )
        return ub_id

    def update_q_id(self, s_id, a_id, r_id, s_da_next):
        """
        Update Q_ID by fitting to the Bellman target:
        Q_ID(s,a) -> r_ID + gamma * Q_DA(s', policy_DA(s'))
        """
        gamma = self.training_params.gamma

        def loss_fn(params):
            q_estimate = self.q_id_model.apply(params, s_id, a_id)
            next_da_actions = self.policy_da_model.apply(
                self.policy_da_params, s_da_next
            )
            q_da_values = self.q_da_model.apply(
                self.q_da_target_params, s_da_next, next_da_actions
            )
            q_target = r_id + gamma * q_da_values
            return mse_loss(q_estimate, q_target)

        grads = jax.grad(loss_fn)(self.q_id_params)
        updates, self.q_id_opt_state = self.q_id_opt.update(grads, self.q_id_opt_state)
        self.q_id_params = optax.apply_updates(self.q_id_params, updates)

    def update_q_da(self, s_da, a_da, r_da, s_id_next):
        """
        Update Q_DA by fitting to the Bellman target:
        Q_DA(s,a) -> r_DA + gamma * Q_ID(s', policy_ID(s'))
        """
        gamma = self.training_params.gamma

        def loss_fn(params):
            q_da_values = self.q_da_model.apply(params, s_da, a_da)
            next_id_actions = self.policy_id_model.apply(
                self.policy_id_params, s_id_next
            )
            q_id_values = self.q_id_model.apply(
                self.q_id_target_params, s_id_next, next_id_actions
            )
            q_target_da = r_da + gamma * q_id_values
            return mse_loss(q_da_values, q_target_da)

        grads = jax.grad(loss_fn)(self.q_da_params)
        updates, self.q_da_opt_state = self.q_da_opt.update(grads, self.q_da_opt_state)
        self.q_da_params = optax.apply_updates(self.q_da_params, updates)

    def update_policy_da(self, s_da):
        """
        Update Policy_DA by maximizing Q_DA(s, policy_DA(s)).
        """

        def loss_fn(params):
            a_da = self.policy_da_model.apply(params, s_da)
            q_values = self.q_da_model.apply(self.q_da_params, s_da, a_da)
            return -jnp.mean(q_values)

        grads = jax.grad(loss_fn)(self.policy_da_params)
        updates, self.policy_da_opt_state = self.policy_da_opt.update(
            grads, self.policy_da_opt_state
        )
        self.policy_da_params = optax.apply_updates(self.policy_da_params, updates)

    def update_policy_id(self, s_id):
        """
        Update Policy_ID by maximizing Q_ID(s, policy_ID(s)).
        """

        def loss_fn(params):
            a_id = self.policy_id_model.apply(params, s_id)
            q_values = self.q_id_model.apply(self.q_id_params, s_id, a_id)
            return -jnp.mean(q_values)

        grads = jax.grad(loss_fn)(self.policy_id_params)
        updates, self.policy_id_opt_state = self.policy_id_opt.update(
            grads, self.policy_id_opt_state
        )
        self.policy_id_params = optax.apply_updates(self.policy_id_params, updates)

    def update_policy_id_with_penalty(self, states_id):
        """
        Update Policy_ID by maximizing Q_ID(s, policy_ID(s)) with penalty for constraint violations.
        """
        Delta_ti = self.sim_params.Delta_ti
        beta_pump = self.sim_params.beta_pump
        beta_turbine = self.sim_params.beta_turbine
        c_pump_up = self.sim_params.c_pump_up
        c_pump_down = self.sim_params.c_pump_down
        c_turbine_up = self.sim_params.c_turbine_up
        c_turbine_down = self.sim_params.c_turbine_down
        x_min_pump = self.sim_params.x_min_pump
        x_max_pump = self.sim_params.x_max_pump
        x_min_turbine = self.sim_params.x_min_turbine
        x_max_turbine = self.sim_params.x_max_turbine
        Rmax = self.sim_params.Rmax

        def loss_fn(params):
            a_id = self.policy_id_model.apply(params, states_id)
            q_values = self.q_id_model.apply(self.q_id_params, states_id, a_id)

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
            relaxation = 1e2

            # Concatenate all constraints
            # A_total: (batch_size, num_constraints, action_size)
            A_total = jnp.concatenate([A, Aeq, -Aeq], axis=1)
            b_total = jnp.concatenate([b, beq + relaxation, -beq + relaxation], axis=1)

            # Penalty for A * x <= b
            Ax = jnp.einsum(
                "bca,ba->bc", A_total, a_id
            )  # (batch_size, num_constraints)
            penalty_ineq = jnp.maximum(Ax - b_total, 0.0)
            penalty = jnp.sum(penalty_ineq**2) # / batch_size

            # Return negative Q + penalty
            return -jnp.mean(q_values) + penalty

        grads = jax.grad(loss_fn)(self.policy_id_params)
        updates, self.policy_id_opt_state = self.policy_id_opt.update(
            grads, self.policy_id_opt_state
        )
        self.policy_id_params = optax.apply_updates(self.policy_id_params, updates)

    def train(self, da_data, id_data):
        batch_size = self.training_params.batch_size
        num_epochs = self.training_params.num_epochs
        tau = self.training_params.tau

        for epoch in range(num_epochs):
            # Train Q_ID and Policy_ID
            for s_id, a_id, r_id, s_da_next in batch_iter(id_data, batch_size):
                s_id = jnp.array(s_id, dtype=jnp.float32)
                a_id = jnp.array(a_id, dtype=jnp.float32)
                r_id = jnp.array(r_id, dtype=jnp.float32).reshape(-1, 1)
                s_da_next = jnp.array(s_da_next, dtype=jnp.float32)

                # Update Q_ID network
                self.update_q_id(s_id, a_id, r_id, s_da_next)

                # Update Policy_ID network with penalties
                self.update_policy_id_with_penalty(s_id)

            # Train Q_DA and Policy_DA
            for s_da, a_da, r_da, s_id_next in batch_iter(da_data, batch_size):
                s_da = jnp.array(s_da, dtype=jnp.float32)
                a_da = jnp.array(a_da, dtype=jnp.float32)
                r_da = jnp.array(r_da, dtype=jnp.float32).reshape(-1, 1)
                s_id_next = jnp.array(s_id_next, dtype=jnp.float32)

                # Update Q_DA network
                self.update_q_da(s_da, a_da, r_da, s_id_next)

                # Update Policy_DA network
                self.update_policy_da(s_da)

            # Perform soft updates of target networks
            self.q_da_target_params = soft_update(
                self.q_da_target_params, self.q_da_params, tau
            )
            self.q_id_target_params = soft_update(
                self.q_id_target_params, self.q_id_params, tau
            )

            print(f"Epoch {epoch+1}/{num_epochs} completed.")
