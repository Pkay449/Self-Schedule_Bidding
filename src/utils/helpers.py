# src/utils/helpers.py

"""
Helper functions for the Self-Scheduled Bidding project.

This module contains functions for computing weights, sampling prices,
generating scenarios, and building constraints for optimization problems.
"""


import os
import warnings

import jax.numpy as jnp
import matlab.engine
import numpy as np
from jax import vmap
from qpsolvers import available_solvers, solve_qp
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal

from src.config import DATA_PATH

warnings.filterwarnings("ignore")


def badp_weights(T=5, data_dir="BADAP_w_data"):
    """
    Compute weights for day-ahead and intraday price influence using predefined coefficients.

    Parameters
    ----------
    T : int, optional
        Number of forecast time steps. Default is 5.
    D : int, optional
        Number of days in the forecast. Default is 7.
    Season : str, optional
        The season name for selecting the appropriate data files (e.g., "Summer"). Default is "Summer".
    gamma : float, optional
        Discount factor for futures prices. Default is 1.0.

    Returns
    -------
    weights : np.ndarray
        Computed weights for day-ahead and intraday prices as a 2D array.
    """
    # Parameters
    D = 7  # days in forecast
    Season = "Summer"
    gamma = 1.0  # discount factor for futures prices

    # Load data
    # Adjust file paths and variable keys according to your environment and data files

    full_data_dir = os.path.join(DATA_PATH, data_dir)
    beta_da_path = os.path.join(full_data_dir, f"beta_day_ahead_{Season}.mat")
    beta_id_path = os.path.join(full_data_dir, f"beta_intraday_{Season}.mat")
    beta_day_ahead_data = loadmat(beta_da_path)
    beta_intraday_data = loadmat(beta_id_path)
    beta_day_ahead = beta_day_ahead_data["beta_day_ahead"]
    beta_intraday = beta_intraday_data["beta_intraday"]

    # In MATLAB:
    # beta_day_ahead(:,1:8)=[]; and beta_intraday(:,1:8)=[]
    # means we remove the first 8 columns of beta_day_ahead and beta_intraday
    beta_day_ahead = beta_day_ahead[:, 8:]
    beta_intraday = beta_intraday[:, 8:]

    # Initialize arrays
    # einfluss_day has shape (T, 24*D)
    einfluss_day = np.zeros((T, 24 * D))
    # einfluss_intraday has shape (T, 24*4*D) = (T, 96*D)
    einfluss_intraday = np.zeros((T, 96 * D))

    # Compute einfluss_day
    for t in range(T):
        for i in range(24 * D):
            Pt_day = np.zeros(24 * D)
            Pt_day[i] = 1
            Pt_intraday = np.zeros(96 * D)
            # Wt_day_mat and Wt_intraday_mat are used but not saved; can be omitted or kept for clarity
            Wt_day_mat = np.zeros((T, 24))
            Wt_intraday_mat = np.zeros((T, 96))

            for t_strich in range(t, T):
                # In MATLAB: Wt_day is 1x(24*D)* ( (24*D)x24 ) = 1x24
                Wt_day = Pt_day @ beta_day_ahead.T  # shape: (24,)

                # Concatenate Wt_day (1x24), Pt_day (1x(24*D)), and Pt_intraday (1x(96*D))
                # Then multiply by beta_intraday'
                combined_intraday = np.concatenate([Wt_day, Pt_day, Pt_intraday])
                Wt_intraday = combined_intraday @ beta_intraday.T  # shape: (96,)

                # Update Pt_day and Pt_intraday as per code:
                Pt_day = np.concatenate([Wt_day, Pt_day[:-24]])
                Pt_intraday = np.concatenate([Wt_intraday, Pt_intraday[:-96]])
                Wt_day_mat[t_strich, :] = Wt_day
                Wt_intraday_mat[t_strich, :] = Wt_intraday
                einfluss_day[t, i] += (
                    4 * np.sum(np.abs(Wt_day)) + np.sum(np.abs(Wt_intraday))
                ) * (gamma ** (t_strich - t))

    # Compute einfluss_intraday
    for t in range(T):
        for i in range(96 * D):
            Pt_day = np.zeros(24 * D)
            Pt_intraday = np.zeros(96 * D)
            Pt_intraday[i] = 1
            Wt_day_mat = np.zeros((T, 24))
            Wt_intraday_mat = np.zeros((T, 96))

            for t_strich in range(t, T):
                Wt_day = Pt_day @ beta_day_ahead.T
                combined_intraday = np.concatenate([Wt_day, Pt_day, Pt_intraday])
                Wt_intraday = combined_intraday @ beta_intraday.T

                Pt_day = np.concatenate([Wt_day, Pt_day[:-24]])
                Pt_intraday = np.concatenate([Wt_intraday, Pt_intraday[:-96]])
                Wt_day_mat[t_strich, :] = Wt_day
                Wt_intraday_mat[t_strich, :] = Wt_intraday
                einfluss_intraday[t, i] += (
                    4 * np.sum(np.abs(Wt_day)) + np.sum(np.abs(Wt_intraday))
                ) * (gamma ** (t_strich - t))

    weights = np.hstack([einfluss_day, einfluss_intraday])
    weights = np.round(weights, 0)
    return weights


def VRx_weights(phi, Y, weights_lsqlin):
    """
    Compute VRx weights using quadratic programming and kernel-based distance adjustment.

    Parameters:
    phi (ndarray): Feature matrix of shape (N, d), where N is the number of samples and d is the number of features.
    Y (ndarray): Target values of shape (N,) or (N, 1).
    weights_lsqlin (ndarray): Initial weights for the features.

    Returns:
    ndarray: Optimized weights of shape (N,).
    """
    if Y.ndim > 1:
        Y = Y.flatten()

    # Step 1: Ignore constant features
    mask = np.std(phi, axis=0) < 0.1
    weights_lsqlin[mask] = 0

    # Step 2: Compute weighted Euclidean distance
    weighted_diff = (phi - Y) * weights_lsqlin
    dist = np.sqrt(np.sum(weighted_diff**2, axis=1))

    N = phi.shape[0]

    def kernel_dist(h_kernel):
        return np.exp(-((dist / h_kernel) ** 2))

    # Step 4: find h_kernel
    def target_function(h_kernel):
        return abs(np.sum(kernel_dist(h_kernel)) - 2 * np.log(N))

    result = minimize_scalar(target_function, bounds=(1e-5, 1e5), method="bounded")
    h_kernel = result.x
    opt_kernel_dist = kernel_dist(h_kernel)

    # Setup QP problem
    W = np.diag(weights_lsqlin)
    H = phi @ W @ phi.T
    H = 0.5 * (H + H.T)  # ensure symmetric

    # Increase epsilon if needed
    epsilon = 1e-8
    H += epsilon * np.eye(N)

    c = -(phi @ W @ Y)
    A = np.ones((1, N))
    b = np.array([1.0])
    lb = -opt_kernel_dist
    ub = opt_kernel_dist

    # Try quadprog first
    try:
        weights = solve_qp(P=H, q=c, A=A, b=b, lb=lb, ub=ub, solver="quadprog")
    except Exception:
        print("quadprog failed due to positive definiteness. Trying another solver...")

        # Check what other solvers are available
        solvers = available_solvers
        # For example, try 'osqp' if available
        if "osqp" in solvers:
            weights = solve_qp(P=H, q=c, A=A, b=b, lb=lb, ub=ub, solver="osqp")
        else:
            raise RuntimeError("No suitable solver available or all failed.")

    return weights


def sample_price_day(Pt_day, t, Season, data_dir="BADAP_w_data"):
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
    full_data_dir = os.path.join(DATA_PATH, data_dir)
    beta_day_ahead_path = os.path.join(full_data_dir, f"beta_day_ahead_{Season}.mat")
    cov_day_path = os.path.join(full_data_dir, f"cov_day_{Season}.mat")
    DoW_path = os.path.join(full_data_dir, f"DoW_{Season}.mat")
    try:
        # Load required data
        beta_day_ahead_data = loadmat(beta_day_ahead_path)
        cov_day_data = loadmat(cov_day_path)
        DoW_data = loadmat(DoW_path)
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


def sample_price_intraday(Pt_day, Pt_intraday, t, Season, data_dir="BADAP_w_data"):
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
    full_data_dir = os.path.join(DATA_PATH, data_dir)
    beta_day_ahead_path = os.path.join(full_data_dir, f"beta_day_ahead_{Season}.mat")
    cov_day_path = os.path.join(full_data_dir, f"cov_day_{Season}.mat")
    beta_intraday_path = os.path.join(full_data_dir, f"beta_intraday_{Season}.mat")
    cov_intraday_path = os.path.join(full_data_dir, f"cov_intraday_{Season}.mat")
    DoW_path = os.path.join(full_data_dir, f"DoW_{Season}.mat")
    try:
        beta_day_ahead_data = loadmat(beta_day_ahead_path)
        cov_day_data = loadmat(cov_day_path)
        beta_intraday_data = loadmat(beta_intraday_path)
        cov_intraday_data = loadmat(cov_intraday_path)
        DoW_data = loadmat(DoW_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required data file is missing: {e.filename}")

    # Extract variables
    beta_day_ahead_data["beta_day_ahead"]
    cov_day_data["cov_day"]
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


def generate_scenarios(N, T, D, P_day_0, P_intraday_0, Season, seed=None):
    """
    Generate scenarios for daily and intraday price simulations.

    Parameters:
    N (int): Number of scenarios to generate.
    T (int): Number of time steps.
    D (int): Number of days.
    P_day_0 (ndarray): Initial daily prices array of shape (24 * D,).
    P_intraday_0 (ndarray): Initial intraday prices array of shape (96 * D,).
    Season (str): Season indicator for price modeling.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    tuple:
        - sample_P_day_all (ndarray): Simulated daily price scenarios, shape (N, T, 24 * D).
        - sample_P_intraday_all (ndarray): Simulated intraday price scenarios, shape (N, T, 96 * D).
        - Wt_day_mat (ndarray): Daily weight matrix, shape (N, T * 24).
        - Wt_intra_mat (ndarray): Intraday weight matrix, shape (N, T * 96).
    """
    if seed is not None:
        np.random.seed(seed)
    sample_P_day_all = np.zeros((N, T, 24 * D))
    sample_P_intraday_all = np.zeros((N, T, 96 * D))
    Wt_day_mat = np.zeros((N, T * 24))
    Wt_intra_mat = np.zeros((N, T * 96))
    for n in range(N):
        P_day = P_day_0[: 24 * D].copy()
        P_intraday = P_intraday_0[: 96 * D].copy()
        for t_i in range(T):
            sample_P_day_all[n, t_i, :] = P_day
            sample_P_intraday_all[n, t_i, :] = P_intraday
            mu_day, cor_day = sample_price_day(P_day, t_i, Season)
            Wt_day = multivariate_normal.rvs(mean=mu_day, cov=cor_day)

            mu_intraday, cor_intraday = sample_price_intraday(
                np.concatenate([Wt_day, P_day]), P_intraday, t_i, Season
            )
            Wt_intraday = multivariate_normal.rvs(mean=mu_intraday, cov=cor_intraday)

            P_day = np.concatenate([Wt_day, P_day[:-24]])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96]])

            Wt_day_mat[n, t_i * 24 : (t_i + 1) * 24] = Wt_day
            Wt_intra_mat[n, t_i * 96 : (t_i + 1) * 96] = Wt_intraday
    return sample_P_day_all, sample_P_intraday_all, Wt_day_mat, Wt_intra_mat


def compute_weights(eng, phi, Y, weights_vector):
    """
    Compute weights by calling VRx_weights for a given feature matrix, targets,
    and initial weights vector.

    Parameters
    ----------
    eng : object
        A MATLAB engine instance or a placeholder for a required computational environment.
    phi : ndarray
        Feature matrix of shape (N, d).
    Y : ndarray
        Target values of shape (N,).
    weights_vector : ndarray
        Initial weights for the features.

    Returns
    -------
    ndarray
        The computed (optimized) weights.
    """
    weights = VRx_weights(phi, Y, weights_vector)
    return weights


def build_and_solve_intlinprog(eng, f, A, b, Aeq, beq, lb, ub, intcon, options):
    """
    Build and solve a linear mixed-integer problem using the MATLAB 'intlinprog' solver.

    Parameters
    ----------
    eng : object
        A MATLAB engine instance.
    f : ndarray
        Coefficients for the objective function.
    A : ndarray
        Coefficients for the inequality constraints.
    b : ndarray
        Right-hand side for the inequality constraints.
    Aeq : ndarray
        Coefficients for the equality constraints.
    beq : ndarray
        Right-hand side for the equality constraints.
    lb : ndarray
        Lower bound constraints for variables.
    ub : ndarray
        Upper bound constraints for variables.
    intcon : list or ndarray
        Indices of variables that are constrained to be integers.
    options : dict
        Options for the MATLAB 'intlinprog' solver.

    Returns
    -------
    x_opt : ndarray
        Optimal solution found by the solver.
    fval : float
        Objective function value at the solution.
    """
    matlab_f = matlab.double((-f).tolist())  # MATLAB: minimize
    matlab_A = matlab.double(A.tolist())
    matlab_b = matlab.double(b.tolist())
    matlab_Aeq = matlab.double(Aeq.tolist())
    matlab_beq = matlab.double(beq.tolist())
    matlab_lb = matlab.double(lb.tolist())
    matlab_ub = matlab.double(ub.tolist())
    matlab_intcon = matlab.double((intcon + 1).tolist())
    xres, fvalres = eng.intlinprog(
        matlab_f,
        matlab_intcon,
        matlab_A,
        matlab_b,
        matlab_Aeq,
        matlab_beq,
        matlab_lb,
        matlab_ub,
        options,
        nargout=2,
    )
    x_opt = np.array(xres).flatten()
    fval = float(fvalres)
    return x_opt, fval


def linear_constraints_train(
    Delta_ti,
    beta_pump,
    beta_turbine,
    c_pump_up,
    c_pump_down,
    c_turbine_up,
    c_turbine_down,
    R_val,
    x0,
    x_min_pump,
    x_max_pump,
    x_min_turbine,
    x_max_turbine,
    Rmax,
    lk,
    VR_abc_neg,
    VR_abc_pos,
):
    """
    Construct linear constraints for a training or optimization scenario involving
    pumping and turbine operations.

    Parameters
    ----------
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
    R_val : float
        Some reference or initial reservoir value.
    x0 : float
        Initial water volume or state variable.
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
    lk : int
        An index or parameter used for piecewise constraints (e.g., VR_abc_neg / VR_abc_pos).
    VR_abc_neg : ndarray
        Coefficients for negative piecewise constraints.
    VR_abc_pos : ndarray
        Coefficients for positive piecewise constraints.

    Returns
    -------
    A : ndarray
        Combined inequality constraint matrix.
    b : ndarray
        Combined inequality constraint vector.
    Aeq : ndarray
        Combined equality constraint matrix.
    beq : ndarray
        Combined equality constraint vector.
    lb : ndarray
        Variable lower bounds.
    ub : ndarray
        Variable upper bounds.
    """
    A1 = np.hstack(
        [
            -np.eye(96) + np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            Delta_ti * beta_pump * np.eye(96),
            -Delta_ti / beta_turbine * np.eye(96),
            -beta_pump * c_pump_up * np.eye(96),
            beta_pump * c_pump_down * np.eye(96),
            c_turbine_up / beta_turbine * np.eye(96),
            -c_turbine_down / beta_turbine * np.eye(96),
            np.zeros((96, 96 * 4 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b1 = np.zeros(96)
    b1[0] = -R_val

    # A2
    Axh = np.zeros((96, 24))
    for h in range(24):
        Axh[h * 4 : (h + 1) * 4, h] = -1

    A2 = np.hstack(
        [
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            -np.eye(96),
            np.zeros((96, 96 * 8)),
            Axh,
            np.zeros((96, 1)),
        ]
    )
    b2 = np.zeros(96)

    A3 = np.hstack(
        [
            np.zeros((96, 96 * 2)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96 * 6 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b3 = np.zeros(96)
    b3[0] = max(x0, 0)

    A4 = np.hstack(
        [
            np.zeros((96, 96 * 3)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96 * 2)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96 * 4 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b4 = np.zeros(96)
    b4[0] = max(-x0, 0)

    # Stack A1, A2, A3, A4 vertically (row-wise)
    Aeq = np.vstack([A1, A2, A3, A4])

    # Stack b1, b2, b3, b4 vertically (row-wise)
    beq = np.hstack([b1, b2, b3, b4])

    A1 = np.vstack(
        [
            np.hstack(
                [
                    np.zeros((96, 96 * 2)),
                    -np.eye(96),
                    np.zeros((96, 96 * 5)),
                    x_min_pump * np.eye(96),
                    np.zeros((96, 96 * 3 + 24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96 * 2)),
                    np.eye(96),
                    np.zeros((96, 96 * 5)),
                    -x_max_pump * np.eye(96),
                    np.zeros((96, 96 * 3 + 24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96 * 3)),
                    -np.eye(96),
                    np.zeros((96, 96 * 5)),
                    x_min_turbine * np.eye(96),
                    np.zeros((96, 96 * 2 + 24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96 * 3)),
                    np.eye(96),
                    np.zeros((96, 96 * 5)),
                    -x_max_turbine * np.eye(96),
                    np.zeros((96, 96 * 2 + 24)),
                    np.zeros((96, 1)),
                ]
            ),
        ]
    )

    b1 = np.zeros(96 * 4)

    A2 = np.hstack(
        [
            np.zeros((96, 96 * 8)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 96 + 24)),
            np.zeros((96, 1)),
        ]
    )

    # Construct b2
    b2 = np.zeros(96)
    b2[0] = float(x0 > 0)

    # Construct A3
    A3 = np.hstack(
        [
            np.zeros((96, 96 * 9)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 24)),
            np.zeros((96, 1)),
        ]
    )

    # Construct b3
    b3 = np.zeros(96)
    b3[0] = float(x0 < 0)

    A4 = np.hstack(
        [
            np.zeros((96, 96 * 8)),
            np.eye(96),
            np.eye(96),
            np.zeros((96, 2 * 96 + 24)),
            np.zeros((96, 1)),
        ]
    )

    # Construct b4
    b4 = np.ones(96)

    AV_neg = np.zeros((lk - 1, 12 * 96 + 24 + 1))
    AV_neg[:, -1] = 1
    AV_neg[:, 96] = -VR_abc_neg[:, 1].copy()
    AV_neg[:, 4 * 96] = -VR_abc_neg[:, 2].copy()
    bV_neg = VR_abc_neg[:, 0].copy()

    AV_pos = np.zeros((lk - 1, 12 * 96 + 24 + 1))
    AV_pos[:, -1] = 1
    AV_pos[:, 96] = -VR_abc_neg[:, 1].copy()
    AV_pos[:, 3 * 96] = -VR_abc_neg[:, 2].copy()
    bV_pos = VR_abc_pos[:, 0]

    A = np.vstack([A1, A2, A3, A4, AV_neg, AV_pos])
    b = np.concatenate([b1, b2, b3, b4, bV_neg, bV_pos])

    lb = np.concatenate(
        [
            np.zeros(96),
            -np.inf * np.ones(96),
            np.zeros(96 * 10),
            -x_max_turbine * np.ones(24),
            np.full(1, -np.inf),
        ]
    )
    ub = np.concatenate(
        [
            Rmax * np.ones(96),
            np.inf * np.ones(96 * 7),
            np.ones(96 * 4),
            x_max_pump * np.ones(24),
            np.full(1, np.inf),
        ]
    )

    return A, b, Aeq, beq, lb, ub


def build_constraints_single(
    R_val,
    x0,
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
    A1 = jnp.hstack(
        [
            -jnp.eye(96) + jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96)),
            Delta_ti * beta_pump * jnp.eye(96),
            -Delta_ti / beta_turbine * jnp.eye(96),
            -beta_pump * c_pump_up * jnp.eye(96),
            beta_pump * c_pump_down * jnp.eye(96),
            c_turbine_up / beta_turbine * jnp.eye(96),
            -c_turbine_down / beta_turbine * jnp.eye(96),
            jnp.zeros((96, 96 * 4)),
        ]
    )
    b1 = jnp.zeros(96).at[0].set(-R_val)

    # A2
    Axh = jnp.zeros((96, 24))
    for h in range(24):
        Axh = Axh.at[4 * h : 4 * (h + 1), h].set(-1)

    A2 = jnp.hstack(
        [
            jnp.zeros((96, 96)),
            -jnp.eye(96),
            jnp.eye(96),
            -jnp.eye(96),
            jnp.zeros((96, 96 * 8)),
        ]
    )
    b2 = jnp.zeros(96)

    # A3
    A3 = jnp.hstack(
        [
            jnp.zeros((96, 96 * 2)),
            jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96)),
            -jnp.eye(96),
            jnp.eye(96),
            jnp.zeros((96, 96 * 6)),
        ]
    )
    b3 = jnp.zeros(96).at[0].set(jnp.maximum(x0, 0))

    # A4
    A4 = jnp.hstack(
        [
            jnp.zeros((96, 96 * 3)),
            jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96 * 2)),
            -jnp.eye(96),
            jnp.eye(96),
            jnp.zeros((96, 96 * 4)),
        ]
    )
    b4 = jnp.zeros(96).at[0].set(jnp.maximum(-x0, 0))

    Aeq = jnp.vstack([A1, A2, A3, A4])
    beq = jnp.hstack([b1, b2, b3, b4])

    # Constraints for pump and turbine power limits
    A1_pump_turbine = jnp.vstack(
        [
            jnp.hstack(
                [
                    jnp.zeros((96, 96 * 2)),
                    -jnp.eye(96),
                    jnp.zeros((96, 96 * 5)),
                    x_min_pump * jnp.eye(96),
                    jnp.zeros((96, 96 * 3)),
                ]
            ),
            jnp.hstack(
                [
                    jnp.zeros((96, 96 * 2)),
                    jnp.eye(96),
                    jnp.zeros((96, 96 * 5)),
                    -x_max_pump * jnp.eye(96),
                    jnp.zeros((96, 96 * 3)),
                ]
            ),
            jnp.hstack(
                [
                    jnp.zeros((96, 96 * 3)),
                    -jnp.eye(96),
                    jnp.zeros((96, 96 * 5)),
                    x_min_turbine * jnp.eye(96),
                    jnp.zeros((96, 96 * 2)),
                ]
            ),
            jnp.hstack(
                [
                    jnp.zeros((96, 96 * 3)),
                    jnp.eye(96),
                    jnp.zeros((96, 96 * 5)),
                    -x_max_turbine * jnp.eye(96),
                    jnp.zeros((96, 96 * 2)),
                ]
            ),
        ]
    )
    b1_pump_turbine = jnp.zeros(96 * 4)

    # Additional constraints if needed:
    A2_additional = jnp.hstack(
        [
            jnp.zeros((96, 96 * 8)),
            jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96)),
            -jnp.eye(96),
            jnp.zeros((96, 96)),
        ]
    )
    b2_additional = jnp.zeros(96).at[0].set((x0 > 0).astype(jnp.float32))

    A3_additional = jnp.hstack(
        [
            jnp.zeros((96, 96 * 9)),
            jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96)),
            -jnp.eye(96),
        ]
    )
    b3_additional = jnp.zeros(96).at[0].set((x0 < 0).astype(jnp.float32))

    A4_additional = jnp.hstack(
        [
            jnp.zeros((96, 96 * 8)),
            jnp.eye(96),
            jnp.eye(96),
            jnp.zeros((96, 2 * 96)),
        ]
    )
    b4_additional = jnp.ones(96)

    A = jnp.vstack([A1_pump_turbine, A2_additional, A3_additional, A4_additional])
    b = jnp.concatenate([b1_pump_turbine, b2_additional, b3_additional, b4_additional])

    # lb and ub
    lb = jnp.concatenate(
        [
            jnp.zeros(96),
            -jnp.inf * jnp.ones(96),
            jnp.zeros(96 * 10),
        ]
    )

    ub = jnp.concatenate(
        [
            Rmax * jnp.ones(96),
            jnp.inf * jnp.ones(96 * 7),
            jnp.ones(96 * 4),
        ]
    )

    return (
        A.astype(jnp.float32),
        b.astype(jnp.float32),
        Aeq.astype(jnp.float32),
        beq.astype(jnp.float32),
        lb.astype(jnp.float32),
        ub.astype(jnp.float32),
    )


def build_constraints_batch(
    states,
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
    x0 = states[:, 1]  # shape: (batch_size,)

    # Vectorize the single constraint builder
    A, b, Aeq, beq, lb, ub = vmap(
        build_constraints_single,
        in_axes=(
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )(
        R_val,
        x0,
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

    return A, b, Aeq, beq, lb, ub  # Each has shape (batch_size, ...)


def reverse_price_series(price_series, granularity=24):
    """
    Reverse the order of a price series by days.

    Parameters
    ----------
    price_series : ndarray
        A 1D array of prices over multiple days, with a known granularity (e.g., 24).
    granularity : int, optional
        The number of data points per day. Defaults to 24.

    Returns
    -------
    res : ndarray
        The reversed price series, day-wise.
    """
    # Reshape into a 2D array where each row represents a day
    price_series_reshaped = price_series.reshape(-1, granularity)

    # Reverse the order of days
    price_series_reversed = price_series_reshaped[::-1]

    # Flatten back into a 1D array
    res = price_series_reversed.flatten()

    return res
