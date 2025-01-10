# src/BADP/main.py

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

import numpy as np

from data.data_io import load_price_data, load_test_data
from src.BADP.eval import evaluate_policy
from src.BADP.gen_offline_samples import generate_offline_data
from src.BADP.train import train_policy
from src.config import DATA_PATH, SimulationParams

# pickle


def main():

    # 1. Load config & data
    sim_params = SimulationParams()
    np.random.seed(sim_params.seed)
    P_day_mat, P_intraday_mat = load_price_data(sim_params.Season)

    # 2. Train policy (Backward Approximate DP)
    Vt, P_day_state, P_intra_state = train_policy(sim_params, P_day_mat, P_intraday_mat)
    # save to src/BADP/objects
    np.save("src/BADP/objects/model_state/Vt.npy", Vt)
    np.save("src/BADP/objects/model_state/P_day_state.npy", P_day_state)
    np.save("src/BADP/objects/model_state/P_intra_state.npy", P_intra_state)

    # Vt = np.load("src/BADP/objects/model_state/Vt.npy")
    # P_day_state = np.load("src/BADP/objects/model_state/P_day_state.npy")
    # P_intra_state = np.load("src/BADP/objects/model_state/P_intra_state.npy")

    # 3. Generate offline samples
    EV, offline_DA, offline_ID = generate_offline_data(
        sim_params, P_day_mat, P_intraday_mat, Vt, P_day_state, P_intra_state
    )
    # save to DATA_PATH/offline_samples
    offline_DA.to_pickle(os.path.join(DATA_PATH, "offline_samples/offline_DA.pkl"))
    offline_ID.to_pickle(os.path.join(DATA_PATH, "offline_samples/offline_ID.pkl"))

    # 4. Evaluate on test data
    # Load test data
    P_day_test, P_intraday_test = load_test_data()
    evaluate_policy(
        sim_params,
        P_day_test,
        P_intraday_test,
        Vt,
        P_day_state,
        P_intra_state,
        "src/BADP/objects/backtest",
    )


if __name__ == "__main__":
    main()
