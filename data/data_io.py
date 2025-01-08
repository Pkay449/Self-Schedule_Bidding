# data/data_io.py

import os
import numpy as np
from scipy.io import loadmat

def load_price_data(season, data_dir="BADAP_w_data"):
    """
    Loads day-ahead and intraday data for a given season from .mat files.
    """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this file
    full_data_dir = os.path.join(base_dir, data_dir)  # Absolute path to data_dir
    
    P_day_mat = loadmat(os.path.join(full_data_dir, f"P_day_{season}.mat"))
    P_intraday_mat = loadmat(os.path.join(full_data_dir, f"P_intraday_{season}.mat"))
    return P_day_mat["P_day_0"].flatten(), P_intraday_mat["P_intraday_0"].flatten()