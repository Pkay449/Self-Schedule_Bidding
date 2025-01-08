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

from src.config import SimulationParams
from data.data_io import load_price_data
import numpy as np
from src.BADP.train import train_policy

def main():
    
    # 1. Load config & data
    sim_params = SimulationParams()
    np.random.seed(sim_params.seed)
    P_day_mat, P_intraday_mat = load_price_data(sim_params.Season)
    
    # 2. Train policy (Backward Approximate DP)
    Vt, P_day_state, P_intra_state = train_policy(sim_params, P_day_mat, P_intraday_mat)
    # save to src/BADP/train_state
    np.save("src/BADP/train_state/Vt.npy", Vt)
    np.save("src/BADP/train_state/P_day_state.npy", P_day_state)
    np.save("src/BADP/train_state/P_intra_state.npy", P_intra_state)

if __name__ == "__main__":
    main()
