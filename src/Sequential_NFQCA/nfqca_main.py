# src/Sequential_NFQCA/nfqca_main.py

import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get PYTHONPATH from the .env file
pythonpath = os.getenv("PYTHONPATH")

if pythonpath:
    # Append PYTHONPATH to sys.path
    path = os.path.abspath(pythonpath)
    sys.path.append(os.path.abspath(pythonpath))

import warnings

import jax

from data.data_io import load_offline_data, load_test_data
from src.config import DA_DATA_PATH, ID_DATA_PATH, SimulationParams, TrainingParams
from src.Sequential_NFQCA.evaluation.evaluation import eval_learned_policy
from src.Sequential_NFQCA.training.trainer import NFQCA


def nfqca_main():
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
    da_data = load_offline_data(DA_DATA_PATH)
    id_data = load_offline_data(ID_DATA_PATH)

    # Create Results directory if it doesn't exist
    os.makedirs("Results/NFQCA", exist_ok=True)

    # Train the model
    nfqca.train(da_data, id_data)

    # Evaluate the learned policies
    P_day_test, P_intraday_test = load_test_data()
    eval_learned_policy(
        policy_id_model=nfqca.policy_id_model,
        policy_da_model=nfqca.policy_da_model,
        policy_id_params=nfqca.policy_id_params,
        policy_da_params=nfqca.policy_da_params,
        sim_params=sim_params,
        P_day_0=P_day_test,
        P_intraday_0=P_intraday_test,
        save_path="src/Sequential_NFQCA/objects/backtest",
    )


if __name__ == "__main__":
    nfqca_main()
