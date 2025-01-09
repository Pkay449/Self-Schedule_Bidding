# src/config.py

from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
import numpy as np

@dataclass
class SimulationParams:
    Season: str = "Summer"
    length_R: int = 5
    seed: int = 2
    D: int = 7  # days in forecast
    Rmax: float = 100.0

    # Ramp times (in hours)
    t_ramp_pump_up: float = 2 / 60
    t_ramp_pump_down: float = 2 / 60
    t_ramp_turbine_up: float = 2 / 60
    t_ramp_turbine_down: float = 2 / 60

    # Grid fees and time deltas
    c_grid_fee: float = 5 / 4
    Delta_ti: float = 0.25
    Delta_td: float = 1.0

    # Q-learning parameters
    Q_mult: float = 1.2
    Q_fix: float = 3
    Q_start_pump: float = 15
    Q_start_turbine: float = 15

    # Pump and turbine parameters
    beta_pump: float = 0.9
    beta_turbine: float = 0.9

    x_max_pump: float = 10.0
    x_min_pump: float = 5.0
    x_max_turbine: float = 10.0
    x_min_turbine: float = 5.0

    # Derived parameters
    R_vec: np.ndarray = field(default_factory=lambda: np.linspace(0, 100.0, 5))
    x_vec: np.ndarray = field(default_factory=lambda: np.array([-10, 0, 10]))

    c_pump_up: float = 2 / 60 / 2
    c_pump_down: float = 2 / 60 / 2
    c_turbine_up: float = 2 / 60 / 2
    c_turbine_down: float = 2 / 60 / 2
    
    N : int = 50 # Number of scenarios
    M : int = 10 # Number of iterations
    T : int = 100 # Optimization horizon
    
    
# PATHS
load_dotenv()
pythonpath = os.getenv("PYTHONPATH")

# Project root directory
ROOT_PATH = os.path.abspath(pythonpath)

# Data directory (root/data)
DATA_PATH = os.path.join(ROOT_PATH, "data")
    