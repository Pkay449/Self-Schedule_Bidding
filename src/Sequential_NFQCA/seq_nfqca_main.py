import os
import pickle as pkl
import warnings
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax import linen as nn
from jax import vmap
from qpsolvers import available_solvers, solve_qp
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull
from scipy.stats import multivariate_normal
from tqdm import tqdm

warnings.filterwarnings("ignore")

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
    
def sample_price_day(Pt_day, t, Season):
    """
    Compute the expected values (mu_P) and covariance matrix (cov_P)
    of day-ahead prices given current observed prices and the current stage.

    Parameters
    ----------
    Pt_day : np.ndarray
        Current observed day-ahead prices as a 1D array (row vector).
    t : int
        Current time stage (1-based index as in MATLAB).
    Season : str
        The season name (e.g., 'Summer').

    Returns
    -------
    mu_P : np.ndarray
        The expected values of day-ahead prices as a 1D array.
    cov_P : np.ndarray
        The covariance matrix of day-ahead prices.
    """
    try:
        # Load required data
        beta_day_ahead_data = loadmat(f"Data/beta_day_ahead_{Season}.mat")
        cov_day_data = loadmat(f"Data/cov_day_{Season}.mat")
        DoW_data = loadmat(f"Data/DoW_{Season}.mat")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required data file is missing: {e.filename}")

    # Extract variables from loaded data
    beta_day_ahead = beta_day_ahead_data["beta_day_ahead"]  # Shape depends on data
    cov_day = cov_day_data["cov_day"]  # Covariance matrix
    # We assume DoW_P0 is stored in DoW_data. The MATLAB code uses DoW_P0 and DoW.
    # Typically, "load(strcat('Data\DoW_',Season,'.mat'))" would load something like DoW_P0.
    # Check the .mat file for the exact variable name.
    # We'll assume it contains a variable DoW_P0. If it's different, rename accordingly.
    DoW_P0 = DoW_data["DoW_P0"].item()  # Assuming it's stored as a scalar

    # Construct day-of-week vector
    DOW = np.zeros(7)
    # MATLAB: DOW(1+mod(t+DoW_P0-1,7))=1;
    # Python is zero-based, but the logic is the same. Just compute the index.
    dow_index = int((t + DoW_P0 - 1) % 7)
    DOW[dow_index] = 1

    # Q = [1, DOW, Pt_day]
    Q = np.concatenate(([1], DOW, Pt_day))

    # In Python: Q (1D array) and beta_day_ahead (2D array)
    # Need to ensure dimensions align. Q.shape: (1+7+(24*D),) and beta_day_ahead: let's assume it matches dimensions.
    mu_P = Q @ beta_day_ahead.T  # Result: 1D array of mu values

    # cov_P is just read from the file
    cov_P = cov_day

    return mu_P, cov_P

def sample_price_intraday(Pt_day, Pt_intraday, t, Season):
    """
    Compute the expected values (mu_P) and covariance matrix (cov_P) of intraday prices
    given current observed day-ahead and intraday prices and the current stage.

    Parameters
    ----------
    Pt_day : np.ndarray
        Current observed day-ahead prices as a 1D array (row vector).
    Pt_intraday : np.ndarray
        Current observed intraday prices as a 1D array (row vector).
    t : int
        Current time stage (1-based index as in MATLAB).
    Season : str
        The season name (e.g., 'Summer').

    Returns
    -------
    mu_P : np.ndarray
        The expected values of intraday prices as a 1D array.
    cov_P : np.ndarray
        The covariance matrix of intraday prices.
    """

    # Load required data
    try:
        beta_day_ahead_data = loadmat(f"Data/beta_day_ahead_{Season}.mat")
        cov_day_data = loadmat(f"Data/cov_day_{Season}.mat")
        beta_intraday_data = loadmat(f"Data/beta_intraday_{Season}.mat")
        cov_intraday_data = loadmat(f"Data/cov_intraday_{Season}.mat")
        DoW_data = loadmat(f"Data/DoW_{Season}.mat")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required data file is missing: {e.filename}")

    # Extract variables
    beta_day_ahead = beta_day_ahead_data["beta_day_ahead"]
    cov_day = cov_day_data["cov_day"]
    beta_intraday = beta_intraday_data["beta_intraday"]
    cov_intraday = cov_intraday_data["cov_intraday"]
    # We assume DoW_P0 is in DoW_data with variable name 'DoW_P0'
    DoW_P0 = DoW_data["DoW_P0"].item()

    # Construct DOW vector
    DOW = np.zeros(7)
    dow_index = int((t + DoW_P0 - 1) % 7)
    DOW[dow_index] = 1

    # Q = [1, DOW, Pt_intraday, Pt_day]
    Q = np.concatenate(([1], DOW, Pt_intraday, Pt_day))

    # mu_P = Q * beta_intraday'
    mu_P = Q @ beta_intraday.T

    # cov_P = cov_intraday
    cov_P = cov_intraday

    return mu_P, cov_P

def build_constraints_single(
    R_val, x0,
    Delta_ti, beta_pump, beta_turbine,
    c_pump_up, c_pump_down,
    c_turbine_up, c_turbine_down,
    x_min_pump, x_max_pump,
    x_min_turbine, x_max_turbine,
    Rmax
):
    """
    Build constraints for a single-step optimization problem involving reservoir state.

    This function uses JAX arrays (jnp) for demonstration, but the structure is similar
    to NumPy-based approaches.

    Parameters
    ----------
    R_val : float
        Some reference or initial reservoir value.
    x0 : float
        Initial water volume or state variable.
    Delta_ti : float
        Time increment or resolution.
    beta_pump : float
        Pump efficiency factor.
    beta_turbine : float
        Turbine efficiency factor.
    c_pump_up : float
        Pump cost factor (up).
    c_pump_down : float
        Pump cost factor (down).
    c_turbine_up : float
        Turbine cost factor (up).
    c_turbine_down : float
        Turbine cost factor (down).
    x_min_pump : float
        Minimum pumping rate.
    x_max_pump : float
        Maximum pumping rate.
    x_min_turbine : float
        Minimum turbine outflow rate.
    x_max_turbine : float
        Maximum turbine outflow rate.
    Rmax : float
        Maximum reservoir capacity.

    Returns
    -------
    A : jnp.ndarray
        Combined inequality constraint matrix for a single step.
    b : jnp.ndarray
        Combined inequality constraint vector for a single step.
    Aeq : jnp.ndarray
        Combined equality constraint matrix for a single step.
    beq : jnp.ndarray
        Combined equality constraint vector for a single step.
    lb : jnp.ndarray
        Variable lower bounds.
    ub : jnp.ndarray
        Variable upper bounds.
    """
    # A1
    A1 = jnp.hstack([
        -jnp.eye(96) + jnp.diag(jnp.ones(95), -1),
        jnp.zeros((96, 96)),
        Delta_ti * beta_pump * jnp.eye(96),
        -Delta_ti / beta_turbine * jnp.eye(96),
        -beta_pump * c_pump_up * jnp.eye(96),
        beta_pump * c_pump_down * jnp.eye(96),
        c_turbine_up / beta_turbine * jnp.eye(96),
        -c_turbine_down / beta_turbine * jnp.eye(96),
        jnp.zeros((96, 96 * 4)),
    ])
    b1 = jnp.zeros(96).at[0].set(-R_val)

    # A2
    Axh = jnp.zeros((96, 24))
    for h in range(24):
        Axh = Axh.at[4 * h:4 * (h + 1), h].set(-1)

    A2 = jnp.hstack([
        jnp.zeros((96, 96)),
        -jnp.eye(96),
        jnp.eye(96),
        -jnp.eye(96),
        jnp.zeros((96, 96 * 8)),
    ])
    b2 = jnp.zeros(96)

    # A3
    A3 = jnp.hstack([
        jnp.zeros((96, 96 * 2)),
        jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
        jnp.zeros((96, 96)),
        -jnp.eye(96),
        jnp.eye(96),
        jnp.zeros((96, 96 * 6)),
    ])
    b3 = jnp.zeros(96).at[0].set(jnp.maximum(x0, 0))

    # A4
    A4 = jnp.hstack([
        jnp.zeros((96, 96 * 3)),
        jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
        jnp.zeros((96, 96 * 2)),
        -jnp.eye(96),
        jnp.eye(96),
        jnp.zeros((96, 96 * 4)),
    ])
    b4 = jnp.zeros(96).at[0].set(jnp.maximum(-x0, 0))

    Aeq = jnp.vstack([A1, A2, A3, A4])
    beq = jnp.hstack([b1, b2, b3, b4])

    # Constraints for pump and turbine power limits
    A1_pump_turbine = jnp.vstack([
        jnp.hstack([
            jnp.zeros((96, 96 * 2)),
            -jnp.eye(96),
            jnp.zeros((96, 96 * 5)),
            x_min_pump * jnp.eye(96),
            jnp.zeros((96, 96 * 3)),
        ]),
        jnp.hstack([
            jnp.zeros((96, 96 * 2)),
            jnp.eye(96),
            jnp.zeros((96, 96 * 5)),
            -x_max_pump * jnp.eye(96),
            jnp.zeros((96, 96 * 3)),
        ]),
        jnp.hstack([
            jnp.zeros((96, 96 * 3)),
            -jnp.eye(96),
            jnp.zeros((96, 96 * 5)),
            x_min_turbine * jnp.eye(96),
            jnp.zeros((96, 96 * 2)),
        ]),
        jnp.hstack([
            jnp.zeros((96, 96 * 3)),
            jnp.eye(96),
            jnp.zeros((96, 96 * 5)),
            -x_max_turbine * jnp.eye(96),
            jnp.zeros((96, 96 * 2)),
        ]),
    ])
    b1_pump_turbine = jnp.zeros(96 * 4)

    # Additional constraints if needed:
    A2_additional = jnp.hstack([
        jnp.zeros((96, 96 * 8)),
        jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
        jnp.zeros((96, 96)),
        -jnp.eye(96),
        jnp.zeros((96, 96)),
    ])
    b2_additional = jnp.zeros(96).at[0].set((x0 > 0).astype(jnp.float32))

    A3_additional = jnp.hstack([
        jnp.zeros((96, 96 * 9)),
        jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
        jnp.zeros((96, 96)),
        -jnp.eye(96),
    ])
    b3_additional = jnp.zeros(96).at[0].set((x0 < 0).astype(jnp.float32))

    A4_additional = jnp.hstack([
        jnp.zeros((96, 96 * 8)),
        jnp.eye(96),
        jnp.eye(96),
        jnp.zeros((96, 2 * 96)),
    ])
    b4_additional = jnp.ones(96)

    A = jnp.vstack([A1_pump_turbine, A2_additional, A3_additional, A4_additional])
    b = jnp.concatenate([b1_pump_turbine, b2_additional, b3_additional, b4_additional])

    # lb and ub
    lb = jnp.concatenate([
        jnp.zeros(96),
        -jnp.inf * jnp.ones(96),
        jnp.zeros(96 * 10),
    ])

    ub = jnp.concatenate([
        Rmax * jnp.ones(96),
        jnp.inf * jnp.ones(96 * 7),
        jnp.ones(96 * 4),
    ])

    return (
        A.astype(jnp.float32),
        b.astype(jnp.float32),
        Aeq.astype(jnp.float32),
        beq.astype(jnp.float32),
        lb.astype(jnp.float32),
        ub.astype(jnp.float32)
    )


def build_constraints_batch(
    states,
    Delta_ti, beta_pump, beta_turbine,
    c_pump_up, c_pump_down,
    c_turbine_up, c_turbine_down,
    x_min_pump, x_max_pump,
    x_min_turbine, x_max_turbine,
    Rmax
):
    """
    Vectorized building of constraints for multiple states in a batch.

    Parameters
    ----------
    states : ndarray
        A 2D array where each row represents a state [R_val, x0].
    Delta_ti : float
        Time increment or resolution.
    beta_pump : float
        Pump efficiency factor.
    beta_turbine : float
        Turbine efficiency factor.
    c_pump_up : float
        Pump cost factor (up).
    c_pump_down : float
        Pump cost factor (down).
    c_turbine_up : float
        Turbine cost factor (up).
    c_turbine_down : float
        Turbine cost factor (down).
    x_min_pump : float
        Minimum pumping rate.
    x_max_pump : float
        Maximum pumping rate.
    x_min_turbine : float
        Minimum turbine outflow rate.
    x_max_turbine : float
        Maximum turbine outflow rate.
    Rmax : float
        Maximum reservoir capacity.

    Returns
    -------
    tuple
        A, b, Aeq, beq, lb, ub for each state in states, each of shape (batch_size, ...).
    """
    R_val = states[:, 0]  # shape: (batch_size,)
    x0 = states[:, 1]     # shape: (batch_size,)

    # Vectorize the single constraint builder
    A, b, Aeq, beq, lb, ub = vmap(
        build_constraints_single,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None)
    )(R_val, x0, Delta_ti, beta_pump, beta_turbine,
       c_pump_up, c_pump_down, c_turbine_up, c_turbine_down,
       x_min_pump, x_max_pump, x_min_turbine, x_max_turbine,
       Rmax)

    return A, b, Aeq, beq, lb, ub  # Each has shape (batch_size, ...)


# %% [markdown]
# # Our Approach - NFQCA
#
# The NFQCA (Neural Fitted Q-Continuous Action) approach builds upon the BAPD baseline by introducing improved action-space modeling and constraint handling through Quadratic Programming (QP) projections and policy regularization. Below is a brief description of key elements of our approach:
#
# #### 1. **QP Projection (`qp_projection`)**
# - Ensures that raw actions generated by the policy network respect predefined bounds and constraints.
# - Solves a QP to minimize the distance between raw actions and feasible actions while enforcing equality, inequality, and bound constraints.
# - Includes relaxation parameters for numerical stability when enforcing constraints.
#
# #### 2. **Offline Data Loading**
# - The `load_offline_data` function loads the offline data (day-ahead and intraday) for supervised learning and reinforcement learning.
# - These datasets consist of state-action-reward-next-state tuples essential for training and evaluating Q-functions and policies.
#
# #### 3. **Neural Network Models**
# - **QNetwork**: Predicts Q-values for state-action pairs using a feedforward network, enabling the learning of value functions for day-ahead (`Q_DA`) and intraday (`Q_ID`) markets.
# - **PolicyDA**: Represents the day-ahead policy, constrained to lie within specific bounds using a sigmoid transformation.
# - **PolicyID**: Defines the intraday policy with complex constraints enforced by combining QP-based projections and penalty-based learning mechanisms.
#
# #### 4. **Constraint Penalty Enforcement**
# - The `update_policy_id_with_penalty` function penalizes actions that violate operational constraints during policy optimization.
# - Leverages a batch-constraint generation function (`build_constraints_batch`) for efficient computation.
#
# #### 5. **Dual Policy Optimization**
# - **Day-Ahead Optimization**: Focuses on optimizing actions for strategic, long-term decisions based on day-ahead price forecasts.
# - **Intraday Optimization**: Refines decisions based on real-time intraday price updates, incorporating tighter operational constraints.
#
# #### 6. **Training Process**
# - Updates Q-functions (`Q_DA` and `Q_ID`) using Bellman targets derived from rewards and next-step Q-values.
# - Trains policies (`PolicyDA` and `PolicyID`) to maximize their respective Q-functions while adhering to operational constraints.
# - Uses a soft-update mechanism to stabilize the learning of target networks for both Q-functions.
#
# #### Improvements were made using our NFQCA approach by incorporating:
# - Robust constraint handling.
# - Enhanced action representation through QP projections.
# - Dual policy optimization tailored for hierarchical decision-making.
#
# ### Architecture
#
# ![image-2.png](attachment:image-2.png)

# %% [markdown]
# ### Define Q and Policy networks

# %%
# Bounds for PolicyID
NEG_INF = -1e8
POS_INF = 1e8

sim_params = SimulationParams()

# Network Definitions
class QNetwork(nn.Module):
    """
    A simple Q-network that takes continuous states and actions as inputs
    and outputs a scalar Q-value.
    """

    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action.astype(jnp.float32)], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class PolicyDA(nn.Module):
    """
    Policy network for the Day-Ahead scenario with bounded continuous actions.
    Actions are scaled via a sigmoid to fit within [lb, ub].
    """

    ub: float
    lb: float
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        actions = nn.Dense(self.action_dim)(x)
        # Scale to [lb, ub]
        actions = self.lb + (self.ub - self.lb) * nn.sigmoid(actions)
        return actions


class PolicyID(nn.Module):
    """
    Policy network for the Intraday scenario with QP-based projection to enforce constraints.
    """

    sim_params: SimulationParams  # Include simulation parameters
    lb: jnp.ndarray
    ub: jnp.ndarray
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    def setup(self):
        # Precompute masks as numpy boolean arrays (static)
        self.bounded_mask_np = (self.lb > NEG_INF) & (self.ub < POS_INF)
        self.lower_bounded_mask_np = (self.lb > NEG_INF) & (self.ub == POS_INF)
        self.upper_bounded_mask_np = (self.lb == NEG_INF) & (self.ub < POS_INF)
        self.unbounded_mask_np = (self.lb == NEG_INF) & (self.ub == POS_INF)

        # Convert to JAX boolean arrays
        self.bounded_mask = jnp.array(self.bounded_mask_np, dtype=bool)
        self.lower_bounded_mask = jnp.array(self.lower_bounded_mask_np, dtype=bool)
        self.upper_bounded_mask = jnp.array(self.upper_bounded_mask_np, dtype=bool)
        self.unbounded_mask = jnp.array(self.unbounded_mask_np, dtype=bool)

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the PolicyID network with efficient action bounding.

        Args:
            state: Input state, shape (batch_size, state_dim)

        Returns:
            Projected actions, shape (batch_size, action_dim)
        """
        # 1. Neural Network to generate raw actions
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        raw_actions = nn.Dense(self.action_dim)(x)  # Shape: (batch_size, action_dim)

        # 2. Apply transformations based on mask types using jnp.where

        # a. Fully Bounded: [lb, ub]
        bounded_actions = self.lb + (self.ub - self.lb) * nn.sigmoid(raw_actions)
        actions = jnp.where(self.bounded_mask, bounded_actions, raw_actions)

        # b. Lower Bounded Only: [lb, +inf)
        lower_bounded_actions = self.lb + nn.softplus(raw_actions)
        actions = jnp.where(self.lower_bounded_mask, lower_bounded_actions, actions)

        # c. Upper Bounded Only: (-inf, ub]
        upper_bounded_actions = self.ub - nn.softplus(raw_actions)
        actions = jnp.where(self.upper_bounded_mask, upper_bounded_actions, actions)

        # d. Unbounded: (-inf, +inf) - already set to raw_actions

        return actions


# %% [markdown]
# ### Helper functions for NFQCA


# %%
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


def batch_iter(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
):
    """
    Generator that yields mini-batches of data.
    """
    states, actions, rewards, next_states = data
    N = len(states)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        idx = indices[start : start + batch_size]
        yield (states[idx], actions[idx], rewards[idx], next_states[idx])


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


def soft_update(target_params, online_params, tau=0.005):
    return jax.tree_util.tree_map(
        lambda tp, op: tp * (1 - tau) + op * tau, target_params, online_params
    )


# %% [markdown]
# ### Load data

# %%
# Load day-ahead and intraday data
da_data = load_offline_data("Data/offline_dataset_day_ahead.pkl")
id_data = load_offline_data("Data/offline_dataset_intraday.pkl")

# %% [markdown]
# ### Bounds for Intraday

# %%
lb_id = np.concatenate(
    [
        np.zeros(96),  # bounded
        NEG_INF * np.ones(96),  # unbounded (lower bound ~ -inf)
        np.zeros(96 * 10),  # bounded
    ]
).astype(np.float32)

ub_id = np.concatenate(
    [
        sim_params.Rmax * np.ones(96),  # bounded
        POS_INF * np.ones(96 * 7),  # unbounded (upper bound ~ inf)
        np.ones(96 * 4),  # bounded
    ]
).astype(np.float32)

# After defining lb_id and ub_id
total_masks = (
    ((lb_id > NEG_INF) & (ub_id < POS_INF)).sum()
    + ((lb_id > NEG_INF) & (ub_id == POS_INF)).sum()
    + ((lb_id == NEG_INF) & (ub_id < POS_INF)).sum()
    + ((lb_id == NEG_INF) & (ub_id == POS_INF)).sum()
)

assert (
    total_masks == lb_id.shape[0]
), f"Sum of all masks ({total_masks}) does not equal action_dim ({lb_id.shape[0]})"

# %% [markdown]
# ### Initialize Q and Policy networks

# %%
key = jax.random.PRNGKey(0)
da_key, id_key, pda_key, pid_key = jax.random.split(key, 4)

q_da_model = QNetwork(state_dim=842, action_dim=24)
q_id_model = QNetwork(state_dim=890, action_dim=1152)
policy_da_model = PolicyDA(
    ub=sim_params.x_max_pump, lb=-sim_params.x_max_turbine, state_dim=842, action_dim=24
)
policy_id_model = PolicyID(
    sim_params=sim_params,
    lb=jnp.array(lb_id),
    ub=jnp.array(ub_id),
    state_dim=890,
    action_dim=1152,
)

dummy_s_da = jnp.ones((1, 842))
dummy_a_da = jnp.ones((1, 24), dtype=jnp.float32)
dummy_s_id = jnp.ones((1, 890))
dummy_a_id = jnp.ones((1, 1152), dtype=jnp.float32)

q_da_params = q_da_model.init(da_key, dummy_s_da, dummy_a_da)
q_id_params = q_id_model.init(id_key, dummy_s_id, dummy_a_id)
policy_da_params = policy_da_model.init(pda_key, dummy_s_da)
policy_id_params = policy_id_model.init(pid_key, dummy_s_id)

q_da_target_params = q_da_params
q_id_target_params = q_id_params

# Optimizers
q_learning_rate = 1e-4
policy_learning_rate = 1e-5

q_da_opt = optax.adam(q_learning_rate)
q_id_opt = optax.adam(q_learning_rate)
policy_da_opt = optax.adam(policy_learning_rate)
policy_id_opt = optax.adam(policy_learning_rate)

q_da_opt_state = q_da_opt.init(q_da_params)
q_id_opt_state = q_id_opt.init(q_id_params)
policy_da_opt_state = policy_da_opt.init(policy_da_params)
policy_id_opt_state = policy_id_opt.init(policy_id_params)

gamma = 0.99
batch_size = 256
num_epochs = 50


# %%
@jax.jit
def update_q_id(
    q_id_params,
    q_id_opt_state,
    q_id_target_params,
    q_da_target_params,
    policy_da_params,
    s_id,
    a_id,
    r_id,
    s_da_next,
):
    """
    Update Q_ID by fitting to the Bellman target:
    Q_ID(s,a) -> r_ID + gamma * Q_DA(s', policy_DA(s'))
    """
    next_da_actions = policy_da_model.apply(policy_da_params, s_da_next)
    q_da_values = q_da_model.apply(q_da_target_params, s_da_next, next_da_actions)
    q_target_id = r_id + gamma * q_da_values

    def loss_fn(params):
        q_estimate = q_id_model.apply(params, s_id, a_id)
        return mse_loss(q_estimate, q_target_id), q_estimate

    grads, q_estimate = jax.grad(loss_fn, has_aux=True)(q_id_params)
    updates, q_id_opt_state_new = q_id_opt.update(grads, q_id_opt_state)
    q_id_params_new = optax.apply_updates(q_id_params, updates)
    return q_id_params_new, q_id_opt_state_new, q_estimate


@jax.jit
def update_q_da(
    q_da_params,
    q_da_opt_state,
    q_da_target_params,
    q_id_target_params,
    policy_id_params,
    s_da,
    a_da,
    r_da,
    s_id_next,
):
    """
    Update Q_DA by fitting to the Bellman target:
    Q_DA(s,a) -> r_DA + gamma * Q_ID(s', policy_ID(s'))
    """
    next_id_actions = policy_id_model.apply(policy_id_params, s_id_next)
    q_id_values = q_id_model.apply(q_id_target_params, s_id_next, next_id_actions)
    q_target_da = r_da + gamma * q_id_values

    def loss_fn(params):
        q_da_values = q_da_model.apply(params, s_da, a_da)
        return mse_loss(q_da_values, q_target_da), q_da_values

    grads, q_da_values = jax.grad(loss_fn, has_aux=True)(q_da_params)
    updates, q_da_opt_state_new = q_da_opt.update(grads, q_da_opt_state)
    q_da_params_new = optax.apply_updates(q_da_params, updates)
    return q_da_params_new, q_da_opt_state_new, q_da_values


@jax.jit
def update_policy_da(policy_da_params, policy_da_opt_state, q_da_params, s_da):
    """
    Update Policy_DA by maximizing Q_DA(s, policy_DA(s)).
    """

    def loss_fn(params):
        a_da = policy_da_model.apply(params, s_da)
        q_values = q_da_model.apply(q_da_params, s_da, a_da)
        return -jnp.mean(q_values)

    grads = jax.grad(loss_fn)(policy_da_params)
    updates, policy_da_opt_state_new = policy_da_opt.update(grads, policy_da_opt_state)
    policy_da_params_new = optax.apply_updates(policy_da_params, updates)
    return policy_da_params_new, policy_da_opt_state_new


@jax.jit
def update_policy_id(policy_id_params, policy_id_opt_state, q_id_params, s_id):
    """
    Update Policy_ID by maximizing Q_ID(s, policy_ID(s)).
    """

    def loss_fn(params):
        a_id = policy_id_model.apply(params, s_id)
        q_values = q_id_model.apply(q_id_params, s_id, a_id)
        return -jnp.mean(q_values)

    grads = jax.grad(loss_fn)(policy_id_params)
    updates, policy_id_opt_state_new = policy_id_opt.update(grads, policy_id_opt_state)
    policy_id_params_new = optax.apply_updates(policy_id_params, updates)
    return policy_id_params_new, policy_id_opt_state_new


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
def update_policy_id_with_penalty(
    policy_id_params,
    policy_id_opt_state,
    q_id_params,
    states_id,
    Delta_ti,
    beta_pump,
    beta_turbine,
    c_pump_up,
    c_pump_down,
    c_turbine_up,
    c_turbine_down,
    x_min_pump,
    x_max_pump,
    x_min_turbine,
    x_max_turbine,
    Rmax,
):
    """
    Update Policy_ID by maximizing Q_ID(s, policy_ID(s)) with penalty for constraint violations.
    """

    def loss_fn(params):
        a_id = policy_id_model.apply(params, states_id)
        q_values = q_id_model.apply(q_id_params, states_id, a_id)

        # Compute constraints
        A, b, Aeq, beq, lb, ub = build_constraints_batch(
            states_id,
            Delta_ti,
            beta_pump,
            beta_turbine,
            c_pump_up,
            c_pump_down,
            c_turbine_up,
            c_turbine_down,
            x_min_pump,
            x_max_pump,
            x_min_turbine,
            x_max_turbine,
            Rmax,
        )
        batch_size = A.shape[0]
        action_size = a_id.shape[-1]
        relaxation = 1e2

        # Create identity matrix with batch dimension
        I = jnp.eye(action_size)
        I = jnp.expand_dims(I, axis=0)  # Shape: (1, action_size, action_size)
        I = jnp.tile(
            I, (batch_size, 1, 1)
        )  # Shape: (batch_size, action_size, action_size)

        # # Concatenate all constraints
        A = jnp.concatenate([A, Aeq, -Aeq], axis=1)
        b = jnp.concatenate([b, beq + relaxation, -beq + relaxation], axis=1)

        # Penalty for A * x <= b
        Ax = jnp.einsum("bkc,bc->bk", A, a_id)  # Shape: (batch_size, num_constraints)
        penalty_ineq = jnp.maximum(Ax - b, 0.0)

        # Aggregate penalties
        penalty = jnp.sum(penalty_ineq**2)

        # Total loss
        return -jnp.mean(q_values) + penalty

    grads = jax.grad(loss_fn)(policy_id_params)
    updates, policy_id_opt_state_new = policy_id_opt.update(grads, policy_id_opt_state)
    policy_id_params_new = optax.apply_updates(policy_id_params, updates)
    return policy_id_params_new, policy_id_opt_state_new


# %% [markdown]
# ## Training loop for Multi-NFQCA

# %%
for epoch in range(num_epochs):
    # Train Q_ID and Policy_ID
    for s_id, a_id, r_id, s_da_next in batch_iter(id_data, batch_size):
        # Fetch a batch of intraday (ID) data consisting of:
        # - `s_id`: states for intraday
        # - `a_id`: actions taken in the intraday phase
        # - `r_id`: rewards received during the intraday phase
        # - `s_da_next`: resulting day-ahead states (next step after intraday)

        # Convert the batch to JAX arrays for computation
        s_id = jnp.array(s_id, dtype=jnp.float32)
        a_id = jnp.array(a_id, dtype=jnp.float32)
        r_id = jnp.array(r_id, dtype=jnp.float32).reshape(-1, 1)
        s_da_next = jnp.array(s_da_next, dtype=jnp.float32)

        # Update Q_ID network
        # The Q_ID network is updated multiple times per batch to stabilize learning
        for i in range(5):
            q_id_params, q_id_opt_state, _ = update_q_id(
                q_id_params,  # Current Q_ID parameters
                q_id_opt_state,  # Optimizer state for Q_ID
                q_id_target_params,  # Target Q_ID parameters (soft updated)
                q_da_target_params,  # Target Q_DA parameters for bootstrapping
                policy_da_params,  # Day-ahead policy parameters
                s_id,  # Current intraday states
                a_id,  # Intraday actions
                r_id,  # Intraday rewards
                s_da_next,  # Next day-ahead states
            )

        # Update Policy_ID network with penalties
        # The policy network for intraday decisions is updated by incorporating penalties
        # to reflect operational constraints (e.g., pump/turbine limits)
        policy_id_params, policy_id_opt_state = update_policy_id_with_penalty(
            policy_id_params,  # Current Policy_ID parameters
            policy_id_opt_state,  # Optimizer state for Policy_ID
            q_id_params,  # Updated Q_ID parameters
            s_id,  # Current intraday states
            sim_params.Delta_ti,  # Time interval scaling factor
            sim_params.beta_pump,  # Pump efficiency factor
            sim_params.beta_turbine,  # Turbine efficiency factor
            sim_params.c_pump_up,  # Pumping cost (up)
            sim_params.c_pump_down,  # Pumping cost (down)
            sim_params.c_turbine_up,  # Turbine cost (up)
            sim_params.c_turbine_down,  # Turbine cost (down)
            sim_params.x_min_pump,  # Minimum pump flow rate
            sim_params.x_max_pump,  # Maximum pump flow rate
            sim_params.x_min_turbine,  # Minimum turbine flow rate
            sim_params.x_max_turbine,  # Maximum turbine flow rate
            sim_params.Rmax,  # Maximum reservoir capacity
        )

    # Train Q_DA and Policy_DA
    for s_da, a_da, r_da, s_id_next in batch_iter(da_data, batch_size):
        # Fetch a batch of day-ahead (DA) data consisting of:
        # - `s_da`: states for day-ahead
        # - `a_da`: actions taken in the day-ahead phase
        # - `r_da`: rewards received during the day-ahead phase
        # - `s_id_next`: resulting intraday states (next step after day-ahead)

        # Convert the batch to JAX arrays for computation
        s_da = jnp.array(s_da, dtype=jnp.float32)
        a_da = jnp.array(a_da, dtype=jnp.float32)
        r_da = jnp.array(r_da, dtype=jnp.float32).reshape(-1, 1)
        s_id_next = jnp.array(s_id_next, dtype=jnp.float32)

        # Update Q_DA network
        # The Q_DA network is updated multiple times per batch to stabilize learning
        for i in range(5):
            q_da_params, q_da_opt_state, _ = update_q_da(
                q_da_params,  # Current Q_DA parameters
                q_da_opt_state,  # Optimizer state for Q_DA
                q_da_target_params,  # Target Q_DA parameters (soft updated)
                q_id_target_params,  # Target Q_ID parameters for bootstrapping
                policy_id_params,  # Intraday policy parameters
                s_da,  # Current day-ahead states
                a_da,  # Day-ahead actions
                r_da,  # Day-ahead rewards
                s_id_next,  # Next intraday states
            )

        # Update Policy_DA network
        # The policy network for day-ahead decisions is updated to optimize actions
        policy_da_params, policy_da_opt_state = update_policy_da(
            policy_da_params,  # Current Policy_DA parameters
            policy_da_opt_state,  # Optimizer state for Policy_DA
            q_da_params,  # Updated Q_DA parameters
            s_da,  # Current day-ahead states
        )

    # Perform soft updates of target networks
    # Gradually update target Q networks (Q_DA and Q_ID) to improve stability
    q_da_target_params = soft_update(q_da_target_params, q_da_params)
    q_id_target_params = soft_update(q_id_target_params, q_id_params)

    # Print progress after completing each epoch
    print(f"Epoch {epoch+1}/{num_epochs} completed.")


# %% [markdown]
# ## Evaluating Learned Policies with NFQCA
#
# We evaluate the performance of learned policies (day-ahead and intra-day) under the NFQCA framework using a held-out test dataset. It simulates a day-ahead and intraday market environment while considering operational constraints and energy market dynamics.


# %%
def eval_learned_policy(
    policy_id_model, policy_da_model, policy_id_params, policy_da_params
):
    """
    Evaluate the performance of the learned policies for intraday and day-ahead energy trading.

    This function simulates the energy trading environment using the learned policies and
    evaluates their performance over multiple time steps. It calculates cumulative rewards
    and tracks key operational metrics.

    Parameters
    ----------
    policy_id_model : callable
        Intraday policy model, which takes intraday state and parameters as input and outputs actions.
    policy_da_model : callable
        Day-ahead policy model, which takes day-ahead state and parameters as input and outputs actions.
    policy_id_params : dict or array
        Parameters of the intraday policy model.
    policy_da_params : dict or array
        Parameters of the day-ahead policy model.

    Returns
    -------
    None
        Saves evaluation results, including paths, cumulative rewards, and environment trackers, to disk.
    """
    M = 1  # Number of scenarios to simulate
    T = 30  # Optimization horizon (time steps)
    D = 7  # Days considered for initial price series
    Season = "Summer"  # Season for the test dataset

    Q_mult = 1.2
    Q_fix = 3
    Q_start_pump = 15
    Q_start_turbine = 15

    c_grid_fee = 5 / 4
    Delta_ti = 0.25
    Delta_td = 1.0

    t_ramp_pump_up = 2 / 60
    t_ramp_pump_down = 2 / 60
    t_ramp_turbine_up = 2 / 60
    t_ramp_turbine_down = 2 / 60

    seed = 2
    np.random.seed(seed)

    P_day_mat = loadmat(os.path.join("Data", f"P_day_ahead_test_all.mat"))
    P_intraday_mat = loadmat(os.path.join("Data", f"P_intraday_test_all.mat"))

    P_day_0 = P_day_mat["P_day_0"].flatten()
    P_intraday_0 = P_intraday_mat["P_intraday_0"].flatten()

    R_0 = 0
    x0_0 = 0
    V = np.zeros((M, 1))

    R_path = np.zeros((M, 96 * T))
    x_intraday_path = np.zeros((M, 96 * T))
    P_day_path = np.zeros((M, 96 * T))
    P_intraday_path = np.zeros((M, 96 * T))
    x_pump_path = np.zeros((M, 96 * T))
    x_turbine_path = np.zeros((M, 96 * T))
    y_pump_path = np.zeros((M, 96 * T))
    y_turbine_path = np.zeros((M, 96 * T))
    z_pump_path = np.zeros((M, 96 * T))
    z_turbine_path = np.zeros((M, 96 * T))

    # Enviroment trackers
    storage_track = []
    storage_track.append(R_0)

    for m in range(M):
        R = R_0
        x0 = x0_0
        P_day = P_day_0[: 24 * D].copy()
        P_intraday = P_intraday_0[: 96 * D].copy()

        P_day_sim = P_day_0[: 24 * D].copy()
        P_intraday_sim = P_intraday_0[: 96 * D].copy()

        C = 0
        for t_i in range(T):
            mu_day, _ = sample_price_day(P_day_sim, t_i, Season)
            mu_intraday, _ = sample_price_intraday(
                np.concatenate([mu_day, P_day_sim]), P_intraday_sim, t_i, Season
            )

            P_day_next = np.concatenate([mu_day, P_day[:-24].copy()])
            P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96].copy()])

            # Compute ramping costs for intraday adjustments
            q_pump_up = (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
            q_pump_down = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
            q_turbine_up = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
            q_turbine_down = (
                (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2
            )

            # Get day ahead initial state
            da_state = np.concatenate([[R], [x0], P_day, P_intraday])

            Wt_day = P_day_0[t_i * 24 : (t_i + 1) * 24].copy()
            day_path = np.tile(Wt_day, (4, 1))
            P_day_path[m, t_i * 96 : (t_i + 1) * 96] = day_path.flatten()

            # Get day ahead action from corresponding policy model
            xday_opt = policy_da_model.apply(policy_da_params, da_state)

            # Get initial state for intraday
            id_state = np.concatenate([da_state, xday_opt, Wt_day])

            # Get intraday action from corresponding policy model
            xid_opt = policy_id_model.apply(policy_id_params, id_state)

            # Extract results from x_opt2
            R_opt = xid_opt[:96].copy()
            xhq_opt = xid_opt[96 : 2 * 96].copy()

            Delta_pump_up = xid_opt[4 * 96 : 5 * 96].copy()
            Delta_pump_down = xid_opt[5 * 96 : 6 * 96].copy()
            Delta_turbine_up = xid_opt[6 * 96 : 7 * 96].copy()
            Delta_turbine_down = xid_opt[7 * 96 : 8 * 96].copy()

            x_pump = xid_opt[2 * 96 : 3 * 96].copy()
            x_turbine = xid_opt[3 * 96 : 4 * 96].copy()
            y_pump = xid_opt[8 * 96 : 9 * 96].copy()
            y_turbine = xid_opt[9 * 96 : 10 * 96].copy()
            z_pump = xid_opt[10 * 96 : 11 * 96].copy()
            z_turbine = xid_opt[11 * 96 : 12 * 96].copy()

            # Update paths
            R_path[m, t_i * 96 : (t_i + 1) * 96] = R_opt
            x_intraday_path[m, t_i * 96 : (t_i + 1) * 96] = xhq_opt
            x_pump_path[m, t_i * 96 : (t_i + 1) * 96] = x_pump
            x_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = x_turbine
            y_pump_path[m, t_i * 96 : (t_i + 1) * 96] = y_pump
            y_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = y_turbine
            z_pump_path[m, t_i * 96 : (t_i + 1) * 96] = z_pump
            z_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = z_turbine

            Wt_intraday = P_intraday_0[t_i * 96 : (t_i + 1) * 96].copy()
            P_intraday_path[m, t_i * 96 : (t_i + 1) * 96] = Wt_intraday

            # Update q_ ramps with realized intraday prices
            q_pump_up = (np.abs(Wt_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
            q_pump_down = (np.abs(Wt_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
            q_turbine_up = (
                (np.abs(Wt_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
            )
            q_turbine_down = (
                (np.abs(Wt_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2
            )

            # Update R, x0, P_day, P_intraday, P_day_sim, P_intraday_sim
            R = R_opt[-1].copy()
            x0 = x_pump[-1] - x_turbine[-1]
            P_day = np.concatenate([Wt_day, P_day[:-24].copy()])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96].copy()])
            P_day_sim = np.concatenate([Wt_day, P_day_sim[:-24].copy()])
            P_intraday_sim = np.concatenate([Wt_intraday, P_intraday_sim[:-96].copy()])

            # Update C
            C = (
                C
                - Delta_td * np.dot(Wt_day, xday_opt)
                - np.sum(x_pump) * c_grid_fee
                - Delta_ti * np.dot(Wt_intraday, xhq_opt)
                + np.dot(q_pump_up, Delta_pump_up)
                - np.dot(q_pump_down, Delta_pump_down)
                - np.dot(q_turbine_up, Delta_turbine_up)
                + np.dot(q_turbine_down, Delta_turbine_down)
                - np.sum(z_pump) * Q_start_pump
                - np.sum(z_turbine) * Q_start_turbine
            )

            # UPDATE TRACKERS
            storage_track.append(R)

        V[m] = C

    EV = np.mean(V)
    print(EV)

    # print backtest statistics :
    print("Backtest Statistics:")
    print("Mean Value: ", np.mean(V))
    print("Standard Deviation: ", np.std(V))
    print("Total Reward: ", np.sum(V))

    # save trackers
    # storage_track
    np.save("Results/NFQCA/BACKTEST_storage_track.npy", storage_track)

    # Save paths
    np.save("Results/NFQCA/BACKTEST_R_path.npy", R_path)
    np.save("Results/NFQCA/BACKTEST_x_intraday_path.npy", x_intraday_path)
    np.save("Results/NFQCA/BACKTEST_P_day_path.npy", P_day_path)
    np.save("Results/NFQCA/BACKTEST_P_intraday_path.npy", P_intraday_path)
    np.save("Results/NFQCA/BACKTEST_x_pump_path.npy", x_pump_path)
    np.save("Results/NFQCA/BACKTEST_x_turbine_path.npy", x_turbine_path)
    np.save("Results/NFQCA/BACKTEST_y_pump_path.npy", y_pump_path)
    np.save("Results/NFQCA/BACKTEST_y_turbine_path.npy", y_turbine_path)
    np.save("Results/NFQCA/BACKTEST_z_pump_path.npy", z_pump_path)
    np.save("Results/NFQCA/BACKTEST_z_turbine_path.npy", z_turbine_path)


# %% [markdown]
# ## Results Visualization - NFQCA
#
# The plots below illustrate the performance of the NFQCA approach in managing energy storage and operational constraints. Key elements such as market prices, intraday actions, and pump/turbine operations are visualized, showcasing the model's ability to adapt dynamically to market signals and system constraints. This visualization highlights how the NFQCA policy efficiently balances energy resources while adhering to operational boundaries, improving upon baseline of the BAPD approach.

# %%
eval_learned_policy(
    policy_id_model, policy_da_model, policy_id_params, policy_da_params
)

R_path = np.load("Results/NFQCA/BACKTEST_R_path.npy").ravel()
x_intraday_path = np.load("Results/NFQCA/BACKTEST_x_intraday_path.npy").ravel()
P_day_path = np.load("Results/NFQCA/BACKTEST_P_day_path.npy").ravel()
P_intraday_path = np.load("Results/NFQCA/BACKTEST_P_intraday_path.npy").ravel()
x_pump_path = np.load("Results/NFQCA/BACKTEST_x_pump_path.npy").ravel()
x_turbine_path = np.load("Results/NFQCA/BACKTEST_x_turbine_path.npy").ravel()
y_pump_path = np.load("Results/NFQCA/BACKTEST_y_pump_path.npy").ravel()
y_turbine_path = np.load("Results/NFQCA/BACKTEST_y_turbine_path.npy").ravel()
z_pump_path = np.load("Results/NFQCA/BACKTEST_z_pump_path.npy").ravel()

# Create a figure with multiple subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("Backtest Data Plots", fontsize=16)

# Plot each array in a subplot
data = {
    "R Path": R_path,
    "x Intraday Path": x_intraday_path,
    "P Day Path": P_day_path,
    "P Intraday Path": P_intraday_path,
    "x Pump Path": x_pump_path,
    "x Turbine Path": x_turbine_path,
    "y Pump Path": y_pump_path,
    "y Turbine Path": y_turbine_path,
    "z Pump Path": z_pump_path,
}

# Iterate through data and subplots
for ax, (title, array) in zip(axs.flat, data.items()):
    ax.plot(array)
    ax.set_title(title)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Values")
    ax.grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Results/backtest_plots.png")

# %% [markdown]
# # Alternative Approaches Explored

# %% [markdown]
# In this work, we primarily relied on **penalty-based constraint enforcement** to ensure the actions generated by the policy remained within permissible operational bounds. However, we also investigated other strategies to handle the complex constraints and potentially improve training stability and computational efficiency.
#
# 1. **Projection Methods:**
#    - **Quadratic Programming (QP) Projection:** Given the Intraday policy’s intricate constraints, we explored using a QP-based projection step. After the policy network produced raw actions, these actions would be **projected** onto the feasible set defined by the constraints, guaranteeing operational compliance.
#    - **Advantages:** Projection methods **guarantee** feasibility, preventing actions that violate constraints. This is particularly attractive in real-world scenarios where safety or reliability constraints must be strictly upheld.
#    - **Limitations:** Unfortunately, our experimental results revealed that the QP-based projection incurred **significant computational overhead** and led to **numerical instability**, thus hindering the scalability and convergence of the approach under our problem settings.
#
# 2. **Residual Learning:**
#    - **Motivation:** We noted that our environment’s state dimension **alternates every two steps**. Hence, we initially implemented two NFQCAs (Neural Fitted Q with Continuous Actions): one for the Day-Ahead states and one for the Intraday states. This duplication arose from the distinct input structures required at each stage.
#    - **Potential for a Single NFQCA:** With **residual learning**, we could have designed a single NFQCA architecture that learns Q-values for both Day-Ahead and Intraday states in a unified manner. This approach might leverage **shared representations** across states, reducing the need for maintaining two separate networks.
#    - **Benefits and Challenges:** Residual learning could simplify the overall architecture and potentially **improve generalization** by sharing parameters. However, it also introduces complexity in determining how the residual blocks interact across different state representations. Additional experimentation would be needed to confirm whether this consolidated approach outperforms separate models in practice.
#
# Overall, while **penalty-based methods** and **dual policy optimization** formed the core of our solution, **projection methods** and **residual learning** highlight possible avenues for future exploration. Further research may focus on overcoming the computational and numerical issues identified in the projection approach, as well as investigating whether a single, residual-based NFQCA framework can effectively handle both Day-Ahead and Intraday state dimensions without compromising performance or stability.

# %% [markdown]
#

# %% [markdown]
#
