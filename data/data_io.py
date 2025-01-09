# data/data_io.py

import os
import numpy as np
from scipy.io import loadmat

from src.config import ROOT_PATH, DATA_PATH

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
    
    da_test_path = os.path.join(full_data_dir, "P_day_ahead_test_all.npy")
    id_test_path = os.path.join(full_data_dir, "P_intraday_test_all.npy")
    
    P_day_mat = loadmat(da_test_path)
    P_intraday_mat = loadmat(id_test_path)
    
    P_day_0 = P_day_mat["P_day_0"].flatten()
    P_intraday_0 = P_intraday_mat["P_intraday_0"].flatten()
    
    return P_day_0, P_intraday_0