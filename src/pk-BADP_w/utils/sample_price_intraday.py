import numpy as np
from scipy.io import loadmat


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
    beta_day_ahead_data = loadmat(f"Data/beta_day_ahead_{Season}.mat")
    cov_day_data = loadmat(f"Data/cov_day_{Season}.mat")
    beta_intraday_data = loadmat(f"Data/beta_intraday_{Season}.mat")
    cov_intraday_data = loadmat(f"Data/cov_intraday_{Season}.mat")
    DoW_data = loadmat(f"Data/DoW_{Season}.mat")

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
