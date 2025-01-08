# src/BADP/train.py

"""
Training module for the Self-Scheduled Bidding project.

This module contains the function to train or derive an optimal policy
over a backward pass across T time steps, involving scenario generation,
price sampling, and optimization using integer linear programming.
"""

import warnings
import matlab.engine
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

from src.helpers import (
    VRx_weights,
    badp_weights,
    build_and_solve_intlinprog,
    generate_scenarios,
    linear_constraints_train,
    sample_price_day,
    sample_price_intraday,
)

# Tuple
from typing import Tuple
from src.config import SimulationParams

warnings.filterwarnings("ignore")

def train_policy(sim_params: SimulationParams, 
                p_day_0: np.ndarray, 
                p_intraday_0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train or derive an optimal policy over a backward pass across T time steps.

    This function:
      1. Loads or computes weights for day-ahead pricing (via `badp_weights`).
      2. Generates scenarios for day-ahead and intraday prices (via `generate_scenarios`).
      3. Iterates backward in time (from t = T-1 down to t = 0), and:
         - Samples prices from the generated scenarios.
         - Computes expected next-step prices (day-ahead and intraday).
         - Optionally computes VRx-based approximate value functions for the future step.
         - Builds linear constraints and solves an integer linear program (`intlinprog`) for each
           reservoir level (R) and water flow state (x) combination.
         - Stores the resulting value function in a 4D array `Vt`.
      4. Returns the final value function `Vt` for all states and times, as well as the price states
         used (`P_day_state` and `P_intra_state`).

    Returns
    -------
    Vt : numpy.ndarray
        A 4D array of shape (length_R, 3, N, T+1) representing the value function 
        at different reservoir levels (length_R), states (3), scenarios (N), 
        and time steps (T+1).
    P_day_state : numpy.ndarray
        A 3D array of shape (N, T, D*24) containing the day-ahead price scenarios.
    P_intra_state : numpy.ndarray
        A 3D array of shape (N, T, D*24*4) containing the intraday price scenarios.
    """
    N = sim_params.N # Number of scenarios
    M = sim_params.M # Number of iterations
    T = sim_params.T # Optimization horizon
    
    eng = matlab.engine.start_matlab()

    # Compute day-ahead weighting factors for the entire horizon T
    weights_D_value = badp_weights(T)

    # Set options for the MATLAB intlinprog solver (suppress output)
    intlinprog_options = eng.optimoptions("intlinprog", "display", "off")

    # =====================
    # Backward pass
    # =====================

    # Generate simulated day-ahead and intraday price scenarios
    sample_P_day_all, sample_P_intraday_all, Wt_day_mat, Wt_intra_mat = generate_scenarios(
        N, T, sim_params.D, p_day_0, p_intraday_0, sim_params.Season, seed=sim_params.seed
    )

    # Copy scenario data to separate arrays for clarity
    P_day_state = sample_P_day_all.copy()
    P_intra_state = sample_P_intraday_all.copy()

    # Initialize the value function array:
    # Shape: (number_of_reservoir_levels, 3, number_of_scenarios, T+1)
    Vt = np.zeros((sim_params.length_R, 3, N, T + 1))

    # Progress bar for time steps (outer loop)
    with tqdm(total=T, desc="Time Steps (t_i)", position=0) as pbar_ti:
        # Iterate backward in time
        for t_i in range(T - 1, -1, -1):
            # Extract the current day's prices for all scenarios
            P_day_sample = P_day_state[:, t_i, :].copy()
            P_intraday_sample = P_intra_state[:, t_i, :].copy()

            # If not the last time step, prepare next-step day and intraday prices
            if t_i < T - 1:
                P_day_sample_next = P_day_state[:, t_i + 1, :].reshape(N, sim_params.D * 24)
                P_intraday_sample_next = P_intra_state[:, t_i + 1, :].reshape(N, sim_params.D * 24 * 4)

            # Inner progress bar over scenarios
            with tqdm(total=N, desc=f"t_i={t_i} | N", leave=False, position=1) as pbar_n:
                for n in range(N):
                    # Extract day-ahead and intraday prices for scenario n at time t_i
                    P_day = P_day_sample[n, :].copy()
                    P_intraday = P_intraday_sample[n, :].copy()

                    # Compute expected day-ahead prices for the next stage
                    mu_day, cor_day = sample_price_day(P_day, t_i, sim_params.Season)
                    
                    # If we are not at the final time step, we can use the actual next-step scenario data
                    # or the computed mu_day as the "next-day" starting point
                    if t_i < T-1:
                        P_next_day = P_day_sample_next[n, :].copy()
                        P_next_day = P_next_day[:24]
                    else:
                        P_next_day = mu_day
                    mu_intraday, cor_intraday = sample_price_intraday(
                        np.concatenate([P_next_day, P_day]), P_intraday, t_i, sim_params.Season
                    )
                    # Construct the next-step prices by "rolling" the current day-ahead or intraday series
                    P_day_next = np.concatenate([mu_day, P_day[:-24]])
                    P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96]])
                    # Initialize piecewise linear parameters (lk is the number of vertices in the hull)
                    lk = 2
                    VR_abc_neg = np.zeros((lk - 1, 3))
                    VR_abc_pos = np.zeros((lk - 1, 3))

                    # If not the last time step, compute VRx-based approximate future cost-to-go
                    if t_i < T - 1:
                        
                        # Feature matrix phi and target Y for VRx
                        phi = np.concatenate([P_day_sample_next, P_intraday_sample_next], axis=1)
                        Y = np.concatenate([P_day_next, P_intraday_next])
                        weights = VRx_weights(phi, Y, weights_D_value[int(t_i + 1), :])
                        
                        # Evaluate the next-step value function for each reservoir level using these weights
                        VRx = np.zeros((sim_params.length_R, 3))
                        for i in range(sim_params.length_R):
                            for j in range(3):
                                VRx[i, j] = Vt[i, j, :, t_i + 1].dot(weights)
                        # Build a convex hull over (R, future_value) to approximate the piecewise function
                        hull_input = np.column_stack([sim_params.R_vec.T, VRx[:, 1]])
                        hull = ConvexHull(hull_input)
                        k = hull.vertices
                        k = np.sort(k)[::-1] # Sort hull vertices in descending order of R
                        
                        lk = len(k) # Number of hull vertices
                        VR = VRx[k, :]
                        R_k = sim_params.R_vec[k]

                        # Construct piecewise linear segments based on these hull vertices
                        if lk > 1:
                            VR_abc_neg = np.zeros((lk - 1, 3))
                            VR_abc_pos = np.zeros((lk - 1, 3))
                            for i in range(1, lk):
                                # Negative slope piece
                                VR_abc_neg[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (
                                    R_k[i] - R_k[i - 1]
                                )
                                VR_abc_neg[i - 1, 0] = VR[i, 1] - VR_abc_neg[i - 1, 1] * R_k[i]
                                VR_abc_neg[i - 1, 2] = -(VR[i - 1, 1] - VR[i - 1, 0]) / (
                                    sim_params.x_vec[1] - sim_params.x_vec[0]
                                )

                            for i in range(1, lk):
                                # Positive slope piece
                                VR_abc_pos[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (
                                    R_k[i] - R_k[i - 1]
                                )
                                VR_abc_pos[i - 1, 0] = VR[i, 1] - VR_abc_pos[i - 1, 1] * R_k[i]
                                VR_abc_pos[i - 1, 2] = (VR[i - 1, 1] - VR[i - 1, 2]) / (
                                    sim_params.x_vec[1] - sim_params.x_vec[2]
                                )
                        else:
                            # If only one vertex, no piecewise segments can be formed
                            VR_abc_neg = np.zeros((0, 3))
                            VR_abc_pos = np.zeros((0, 3))

                    # Evaluate the current-step optimization for each reservoir level and water flow state
                    for iR in range(sim_params.length_R):
                        R_val = sim_params.R_vec[iR]
                        for ix in range(len(sim_params.x_vec)):
                            x0 = sim_params.x_vec[ix]

                            # Build the objective function coefficients
                            f = np.zeros(96 * 12 + 24 + 1)
                            f[-1] = 1
                            # Subtract intraday revenue or cost (scaled by Delta_ti)
                            f[96:192] -= sim_params.Delta_ti * mu_intraday
                            f[-25:-1] = -sim_params.Delta_td * mu_day

                            # Compute various ramping or start-up cost terms for the objective
                            q_pump_up = (abs(mu_intraday) / sim_params.Q_mult - sim_params.Q_fix) * sim_params.t_ramp_pump_up / 2
                            q_pump_down = (abs(mu_intraday) * sim_params.Q_mult + sim_params.Q_fix) * sim_params.t_ramp_pump_down / 2
                            q_turbine_up = (
                                (abs(mu_intraday) * sim_params.Q_mult + sim_params.Q_fix) * sim_params.t_ramp_turbine_up / 2
                            )
                            q_turbine_down = (
                                (abs(mu_intraday) / sim_params.Q_mult - sim_params.Q_fix) * sim_params.t_ramp_turbine_down / 2
                            )
                            # Adjust for grid fee, pumping, and turbine costs
                            f[96 * 2 : 96 * 3] -= sim_params.c_grid_fee
                            f[96 * 4 : 96 * 5] += q_pump_up
                            f[96 * 5 : 96 * 6] -= q_pump_down
                            f[96 * 6 : 96 * 7] -= q_turbine_up
                            f[96 * 7 : 96 * 8] += q_turbine_down
                            f[96 * 10 : 96 * 11] -= sim_params.Q_start_pump
                            f[96 * 11 : 96 * 12] -= sim_params.Q_start_turbine
                            
                            # Build linear constraints (A, b) and (Aeq, beq) and bounds (lb, ub)
                            A, b, Aeq, beq, lb, ub = linear_constraints_train(
                                sim_params.Delta_ti,
                                sim_params.beta_pump,
                                sim_params.beta_turbine,
                                sim_params.c_pump_up,
                                sim_params.c_pump_down,
                                sim_params.c_turbine_up,
                                sim_params.c_turbine_down,
                                R_val,
                                x0,
                                sim_params.x_min_pump,
                                sim_params.x_max_pump,
                                sim_params.x_min_turbine,
                                sim_params.x_max_turbine,
                                sim_params.Rmax,
                                lk,
                                VR_abc_neg,
                                VR_abc_pos,
                            )
                            # Identify which variables are integers      
                            intcon = np.arange(8 * 96, 96 * 10)

                            # Solve the integer linear program
                            x_opt, fval = build_and_solve_intlinprog(eng, f, A, b, Aeq, beq, lb, ub, intcon, intlinprog_options)

                            # The solver returns the minimized cost, so we store -fval as the "value"
                            Vt[iR, ix, n, t_i] = -fval

                    # Update scenario progress bar
                    pbar_n.update(1)  

            # Update time-step progress bar
            pbar_ti.update(1)  

    return Vt, P_day_state, P_intra_state
