# config.py

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:
    # Simulation Parameters
    N: int = 50          # Number of scenarios
    M: int = 10          # Number of initial states (if applicable)
    T: int = 30          # Time steps
    Season: str = "Summer"
    length_R: int = 5
    seed: int = 2
    D: int = 7           # Days in forecast
    Rmax: float = 100.0

    # Ramp Parameters
    t_ramp_pump_up: float = 2 / 60
    t_ramp_pump_down: float = 2 / 60
    t_ramp_turbine_up: float = 2 / 60
    t_ramp_turbine_down: float = 2 / 60

    # Cost Parameters
    c_grid_fee: float = 5 / 4
    Delta_ti: float = 0.25
    Delta_td: float = 1.0

    Q_mult: float = 1.2
    Q_fix: float = 3.0
    Q_start_pump: float = 15.0
    Q_start_turbine: float = 15.0

    beta_pump: float = 0.9
    beta_turbine: float = 0.9

    # Action Limits
    x_max_pump: float = 10.0
    x_min_pump: float = 5.0
    x_max_turbine: float = 10.0
    x_min_turbine: float = 5.0

    # Vectors
    R_vec: np.ndarray = field(init=False)
    x_vec: np.ndarray = field(init=False)

    def __post_init__(self):
        self.R_vec = np.linspace(0, self.Rmax, self.length_R)
        self.x_vec = np.array([-self.x_max_turbine, 0, self.x_max_pump])
