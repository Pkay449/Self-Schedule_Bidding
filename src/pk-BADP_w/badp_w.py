import numpy as np
from scipy.io import loadmat
import matlab.engine
import os
import sys

# import local modules
import badp_weights
import sample_price_day
import sample_price_intraday
import VRx_weights

# Setup
eng = matlab.engine.start_matlab()  # start matlab engine
os.chdir(
    os.path.dirname(os.path.abspath(__file__))
)  # change directory to current file directory

# ================================================
# Parameters (previously defined inside the MATLAB function)
# ================================================
length_R = 5
N = 50
T = 3
M = 10
seed = 1
Season = "Summer"

D = 7  # days in forecast
Rmax = 100
intlinprog_options = {
    # Add solver options as needed
    # 'disp': False
}

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
P_day_mat = loadmat(f"Data/P_day_{Season}.mat")
P_intraday_mat = loadmat(f"Data/P_intraday_{Season}.mat")
P_day_0 = P_day_mat["P_day_0"].flatten()
P_intraday_0 = P_intraday_mat["P_intraday_0"].flatten()
weights_D_value = eng.badp_weights(T)
weights_D_value = np.array(weights_D_value)

def compute_EV(N, M, T, Season, length_R, seed):
    pass

if __name__ == "__main__":
    # Compare current code to badp_w.m
    # I saved values in both the Python and MATLAB code to compare them
    # matlab : output.mat
    # python : output.npy
    # load them and compare
    output_mat = loadmat("output.mat")
    output_py = np.load("output.npy", allow_pickle=True)
    print('Matlab output:')
    print(output_mat)
    print()
    print('Python output:')
    print(output_py)