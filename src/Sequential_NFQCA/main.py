# main.py
from functools import partial

import jax
import jax.numpy as jnp
import optax
from config import (
    BATCH_SIZE,
    DA_DATA_PATH,
    GAMMA,
    ID_DATA_PATH,
    NUM_EPOCHS,
    POLICY_LEARNING_RATE,
    Q_LEARNING_RATE,
    TAU,
    sim_params,
)
from src.Sequential_NFQCA.model.policy_da import PolicyDA
from src.Sequential_NFQCA.model.policy_id import PolicyID
from src.Sequential_NFQCA.model.q_network import QNetwork
from src.Sequential_NFQCA.utils.constraints import build_constraints_batch
from src.Sequential_NFQCA.utils.data_loader import batch_iter, load_offline_data
from src.Sequential_NFQCA.utils.evaluation import (
    eval_learned_policy,
    plot_backtest_results,
)
from src.Sequential_NFQCA.utils.losses import mse_loss
from src.Sequential_NFQCA.utils.optimizers import get_optimizer, soft_update


class NFQCA:
    def __init__(self, sim_params):
        self.sim_params = sim_params
        self.initialize_models()
        self.initialize_optimizers()

    def initialize_models(self):
        key = jax.random.PRNGKey(0)
        da_key, id_key, pda_key, pid_key = jax.random.split(key, 4)

        # Initialize Q Networks
        self.q_da_model = QNetwork(state_dim=842, action_dim=24)
        self.q_id_model = QNetwork(state_dim=890, action_dim=1152)

        # Initialize Policy Networks
        self.policy_da_model = PolicyDA(
            ub=self.sim_params.x_max_pump,
            lb=-self.sim_params.x_max_turbine,
            state_dim=842,
            action_dim=24,
        )
        self.policy_id_model = PolicyID(
            sim_params=self.sim_params,
            lb=jnp.array(self.sim_params.x_min_pump),  # Adjust based on actual lb_id
            ub=jnp.array(self.sim_params.x_max_pump),  # Adjust based on actual ub_id
            state_dim=890,
            action_dim=1152,
        )

        # Dummy inputs for initialization
        dummy_s_da = jnp.ones((1, 842))
        dummy_a_da = jnp.ones((1, 24), dtype=jnp.float32)
        dummy_s_id = jnp.ones((1, 890))
        dummy_a_id = jnp.ones((1, 1152), dtype=jnp.float32)

        # Initialize parameters
        self.q_da_params = self.q_da_model.init(da_key, dummy_s_da, dummy_a_da)
        self.q_id_params = self.q_id_model.init(id_key, dummy_s_id, dummy_a_id)
        self.policy_da_params = self.policy_da_model.init(pda_key, dummy_s_da)
        self.policy_id_params = self.policy_id_model.init(pid_key, dummy_s_id)

        # Initialize target networks
        self.q_da_target_params = self.q_da_params
        self.q_id_target_params = self.q_id_params

    def initialize_optimizers(self):
        # Initialize optimizers
        self.q_da_opt = get_optimizer(Q_LEARNING_RATE)
        self.q_id_opt = get_optimizer(Q_LEARNING_RATE)
        self.policy_da_opt = get_optimizer(POLICY_LEARNING_RATE)
        self.policy_id_opt = get_optimizer(POLICY_LEARNING_RATE)

        # Initialize optimizer states
        self.q_da_opt_state = self.q_da_opt.init(self.q_da_params)
        self.q_id_opt_state = self.q_id_opt.init(self.q_id_params)
        self.policy_da_opt_state = self.policy_da_opt.init(self.policy_da_params)
        self.policy_id_opt_state = self.policy_id_opt.init(self.policy_id_params)

    @partial(jax.jit, static_argnums=(0,))
    def update_q_id(
        self,
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
        next_da_actions = self.policy_da_model.apply(policy_da_params, s_da_next)
        q_da_values = self.q_da_model.apply(
            q_da_target_params, s_da_next, next_da_actions
        )
        q_target_id = r_id + GAMMA * q_da_values

        def loss_fn(params):
            q_estimate = self.q_id_model.apply(params, s_id, a_id)
            return mse_loss(q_estimate, q_target_id), q_estimate

        grads, q_estimate = jax.grad(loss_fn, has_aux=True)(q_id_params)
        updates, q_id_opt_state_new = self.q_id_opt.update(grads, q_id_opt_state)
        q_id_params_new = optax.apply_updates(q_id_params, updates)
        return q_id_params_new, q_id_opt_state_new, q_estimate

    @partial(jax.jit, static_argnums=(0,))
    def update_q_da(
        self,
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
        next_id_actions = self.policy_id_model.apply(policy_id_params, s_id_next)
        q_id_values = self.q_id_model.apply(
            q_id_target_params, s_id_next, next_id_actions
        )
        q_target_da = r_da + GAMMA * q_id_values

        def loss_fn(params):
            q_da_values = self.q_da_model.apply(params, s_da, a_da)
            return mse_loss(q_da_values, q_target_da), q_da_values

        grads, q_da_values = jax.grad(loss_fn, has_aux=True)(q_da_params)
        updates, q_da_opt_state_new = self.q_da_opt.update(grads, q_da_opt_state)
        q_da_params_new = optax.apply_updates(q_da_params, updates)
        return q_da_params_new, q_da_opt_state_new, q_da_values

    @partial(jax.jit, static_argnums=(0,))
    def update_policy_da(
        self, policy_da_params, policy_da_opt_state, q_da_params, s_da
    ):
        """
        Update Policy_DA by maximizing Q_DA(s, policy_DA(s)).
        """

        def loss_fn(params):
            a_da = self.policy_da_model.apply(params, s_da)
            q_values = self.q_da_model.apply(q_da_params, s_da, a_da)
            return -jnp.mean(q_values)

        grads = jax.grad(loss_fn)(policy_da_params)
        updates, policy_da_opt_state_new = self.policy_da_opt.update(
            grads, policy_da_opt_state
        )
        policy_da_params_new = optax.apply_updates(policy_da_params, updates)
        return policy_da_params_new, policy_da_opt_state_new

    @partial(jax.jit, static_argnums=(0,))
    def update_policy_id(
        self, policy_id_params, policy_id_opt_state, q_id_params, s_id
    ):
        """
        Update Policy_ID by maximizing Q_ID(s, policy_ID(s)).
        """

        def loss_fn(params):
            a_id = self.policy_id_model.apply(params, s_id)
            q_values = self.q_id_model.apply(q_id_params, s_id, a_id)
            return -jnp.mean(q_values)

        grads = jax.grad(loss_fn)(policy_id_params)
        updates, policy_id_opt_state_new = self.policy_id_opt.update(
            grads, policy_id_opt_state
        )
        policy_id_params_new = optax.apply_updates(policy_id_params, updates)
        return policy_id_params_new, policy_id_opt_state_new

    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
    def update_policy_id_with_penalty(
        self,
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
            a_id = self.policy_id_model.apply(params, states_id)
            q_values = self.q_id_model.apply(q_id_params, states_id, a_id)

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
            I = jnp.tile(
                I, (batch_size, 1, 1)
            )  # Shape: (batch_size, action_size, action_size)

            # Concatenate all constraints
            A = jnp.concatenate([A, Aeq, -Aeq], axis=1)
            b = jnp.concatenate([b, beq + relaxation, -beq + relaxation], axis=1)

            # Penalty for A * x <= b
            Ax = jnp.einsum(
                "bkc,bc->bk", A, a_id
            )  # Shape: (batch_size, num_constraints)
            penalty_ineq = jnp.maximum(Ax - b, 0.0)

            # Aggregate penalties
            penalty = jnp.sum(penalty_ineq**2)

            # Total loss
            return -jnp.mean(q_values) + penalty

        grads = jax.grad(loss_fn)(policy_id_params)
        updates, policy_id_opt_state_new = self.policy_id_opt.update(
            grads, policy_id_opt_state
        )
        policy_id_params_new = optax.apply_updates(policy_id_params, updates)
        return policy_id_params_new, policy_id_opt_state_new

    def train(self, da_data, id_data):
        for epoch in range(NUM_EPOCHS):
            # Train Q_ID and Policy_ID
            for s_id, a_id, r_id, s_da_next in batch_iter(id_data, BATCH_SIZE):
                s_id = jnp.array(s_id, dtype=jnp.float32)
                a_id = jnp.array(a_id, dtype=jnp.float32)
                r_id = jnp.array(r_id, dtype=jnp.float32).reshape(-1, 1)
                s_da_next = jnp.array(s_da_next, dtype=jnp.float32)

                # Update Q_ID network multiple times per batch
                for _ in range(5):
                    self.q_id_params, self.q_id_opt_state, _ = self.update_q_id(
                        self.q_id_params,
                        self.q_id_opt_state,
                        self.q_id_target_params,
                        self.q_da_target_params,
                        self.policy_da_params,
                        s_id,
                        a_id,
                        r_id,
                        s_da_next,
                    )

                # Update Policy_ID network with penalties
                self.policy_id_params, self.policy_id_opt_state = (
                    self.update_policy_id_with_penalty(
                        self.policy_id_params,
                        self.policy_id_opt_state,
                        self.q_id_params,
                        s_id,
                        self.sim_params.Delta_ti,
                        self.sim_params.beta_pump,
                        self.sim_params.beta_turbine,
                        self.sim_params.c_pump_up,
                        self.sim_params.c_pump_down,
                        self.sim_params.c_turbine_up,
                        self.sim_params.c_turbine_down,
                        self.sim_params.x_min_pump,
                        self.sim_params.x_max_pump,
                        self.sim_params.x_min_turbine,
                        self.sim_params.x_max_turbine,
                        self.sim_params.Rmax,
                    )
                )

            # Train Q_DA and Policy_DA
            for s_da, a_da, r_da, s_id_next in batch_iter(da_data, BATCH_SIZE):
                s_da = jnp.array(s_da, dtype=jnp.float32)
                a_da = jnp.array(a_da, dtype=jnp.float32)
                r_da = jnp.array(r_da, dtype=jnp.float32).reshape(-1, 1)
                s_id_next = jnp.array(s_id_next, dtype=jnp.float32)

                # Update Q_DA network multiple times per batch
                for _ in range(5):
                    self.q_da_params, self.q_da_opt_state, _ = self.update_q_da(
                        self.q_da_params,
                        self.q_da_opt_state,
                        self.q_da_target_params,
                        self.q_id_target_params,
                        self.policy_id_params,
                        s_da,
                        a_da,
                        r_da,
                        s_id_next,
                    )

                # Update Policy_DA network
                self.policy_da_params, self.policy_da_opt_state = self.update_policy_da(
                    self.policy_da_params,
                    self.policy_da_opt_state,
                    self.q_da_params,
                    s_da,
                )

            # Perform soft updates of target networks
            self.q_da_target_params = soft_update(
                self.q_da_target_params, self.q_da_params, TAU
            )
            self.q_id_target_params = soft_update(
                self.q_id_target_params, self.q_id_params, TAU
            )

            # Print progress
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed.")

    def evaluate(self):
        eval_learned_policy(
            self.policy_id_model,
            self.policy_da_model,
            self.policy_id_params,
            self.policy_da_params,
            "src/Sequential_NFQCA/objects/backtest",
        )


if __name__ == "__main__":
    # Load data
    da_data = load_offline_data(DA_DATA_PATH)
    id_data = load_offline_data(ID_DATA_PATH)

    # Initialize NFQCA model
    nfqca = NFQCA(sim_params)

    # Train the model
    nfqca.train(da_data, id_data)

    # Evaluate the learned policies
    nfqca.evaluate()
    plot_backtest_results("src/Sequential_NFQCA/objects/backtest")
