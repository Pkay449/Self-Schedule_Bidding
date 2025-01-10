# src/Sequential_NFQCA/main.py

import os
import jax
import jax.numpy as jnp
from src.config import SimulationParams, TrainingParams
from src.Sequential_NFQCA.utils.data_loader import load_offline_data
from src.Sequential_NFQCA.training.trainer import NFQCA
from src.Sequential_NFQCA.evaluation.evaluation import eval_learned_policy
import warnings

def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Initialize simulation and training parameters
    sim_params = SimulationParams()
    training_params = TrainingParams()

    # Initialize random key
    key = jax.random.PRNGKey(sim_params.seed)

    # Initialize NFQCA model
    nfqca = NFQCA(sim_params, training_params, key)

    # Load offline data
    da_data = load_offline_data("Data/offline_dataset_day_ahead.pkl")
    id_data = load_offline_data("Data/offline_dataset_intraday.pkl")

    # Create Results directory if it doesn't exist
    os.makedirs("Results/NFQCA", exist_ok=True)

    # Train the model
    nfqca.train(da_data, id_data)

    # Evaluate the learned policies
    eval_learned_policy(
        policy_id_model=nfqca.policy_id_model,
        policy_da_model=nfqca.policy_da_model,
        policy_id_params=nfqca.policy_id_params,
        policy_da_params=nfqca.policy_da_params,
        sim_params=sim_params
    )

if __name__ == "__main__":
    main()
