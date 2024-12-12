import numpy as np
import pandas as pd
import matplotlib
from scipy.io import loadmat
import matlab.engine
import os

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Change working directory to the current file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_config(season="Summer"):
    """
    Load configuration settings and initialize parameters.
    """
    # Parameters
    length_R = 5
    N = 50
    T = 3
    M = 10
    seed = 1
    Rmax = 100
    D = 7  # days in forecast

    # Ramp times, efficiencies, fees, etc.
    t_ramp_pump_up = 2 / 60
    t_ramp_pump_down = 2 / 60
    t_ramp_turbine_up = 2 / 60
    t_ramp_turbine_down = 2 / 60

    c_grid_fee = 5 / 4
    Delta_ti = 0.25
    Delta_td = 1

    Q_mult = 1.2
    Q_fix = 3
    Q_start_pump = 15
    Q_start_turbine = 15

    beta_pump = 0.9
    beta_turbine = 0.9

    x_max_pump = 10
    x_min_pump = 5
    x_max_turbine = 10
    x_min_turbine = 5

    R_vec = np.linspace(0, Rmax, length_R)
    x_vec = np.array([-x_max_turbine, 0, x_max_pump])

    c_pump_up = t_ramp_pump_up / 2
    c_pump_down = t_ramp_pump_down / 2
    c_turbine_up = t_ramp_turbine_up / 2
    c_turbine_down = t_ramp_turbine_down / 2

    # Load data
    P_day_mat = loadmat(f"Data/P_day_{season}.mat")
    P_intraday_mat = loadmat(f"Data/P_intraday_{season}.mat")
    P_day_0 = P_day_mat["P_day_0"].flatten()
    P_intraday_0 = P_intraday_mat["P_intraday_0"].flatten()

    # Compute weights using MATLAB engine
    weights_D_value = eng.badp_weights(T)
    weights_D_value = np.array(weights_D_value)

    # Return all variables as a dictionary
    return {
        "length_R": length_R,
        "N": N,
        "T": T,
        "M": M,
        "seed": seed,
        "Rmax": Rmax,
        "D": D,
        "t_ramp_pump_up": t_ramp_pump_up,
        "t_ramp_pump_down": t_ramp_pump_down,
        "t_ramp_turbine_up": t_ramp_turbine_up,
        "t_ramp_turbine_down": t_ramp_turbine_down,
        "c_grid_fee": c_grid_fee,
        "Delta_ti": Delta_ti,
        "Delta_td": Delta_td,
        "Q_mult": Q_mult,
        "Q_fix": Q_fix,
        "Q_start_pump": Q_start_pump,
        "Q_start_turbine": Q_start_turbine,
        "beta_pump": beta_pump,
        "beta_turbine": beta_turbine,
        "x_max_pump": x_max_pump,
        "x_min_pump": x_min_pump,
        "x_max_turbine": x_max_turbine,
        "x_min_turbine": x_min_turbine,
        "R_vec": R_vec,
        "x_vec": x_vec,
        "c_pump_up": c_pump_up,
        "c_pump_down": c_pump_down,
        "c_turbine_up": c_turbine_up,
        "c_turbine_down": c_turbine_down,
        "P_day_0": P_day_0,
        "P_intraday_0": P_intraday_0,
        "weights_D_value": weights_D_value,
    }
