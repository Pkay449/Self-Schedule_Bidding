# data/data_io.py

import os
import pickle as pkl

import numpy as np
from scipy.io import loadmat

from src.config import DATA_PATH


def load_price_data(season, data_dir="BADAP_w_data"):
    """
    Loads day-ahead and intraday data for a given season from .mat files.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this file
    full_data_dir = os.path.join(base_dir, data_dir)  # Absolute path to data_dir

    P_day_mat = loadmat(os.path.join(full_data_dir, f"P_day_{season}.mat"))
    P_intraday_mat = loadmat(os.path.join(full_data_dir, f"P_intraday_{season}.mat"))
    return P_day_mat["P_day_0"].flatten(), P_intraday_mat["P_intraday_0"].flatten()


def load_test_data():
    # Loads data from DATA_PATH\test_data
    full_data_dir = os.path.join(DATA_PATH, "test_data")

    da_test_path = os.path.join(full_data_dir, "P_day_ahead_test_all.mat")
    id_test_path = os.path.join(full_data_dir, "P_intraday_test_all.mat")

    P_day_mat = loadmat(da_test_path)
    P_intraday_mat = loadmat(id_test_path)

    P_day_0 = P_day_mat["P_day_0"].flatten()
    P_intraday_0 = P_intraday_mat["P_intraday_0"].flatten()

    return P_day_0, P_intraday_0


def load_offline_data(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads offline dataset from a pickle file. The file is expected to contain a dictionary
    with keys: "state", "action", "reward", "next_state". Each value should be a Series
    or array-like from which we can extract arrays.

    Returns:
        states, actions, rewards, next_states
    """
    with open(path, "rb") as f:
        df = pkl.load(f)
    states = np.stack(df["state"].values)
    actions = np.stack(df["action"].values)
    rewards = df["reward"].values
    next_states = np.stack(df["next_state"].values)
    return states, actions, rewards, next_states
