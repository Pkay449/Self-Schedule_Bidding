import numpy as np
from scipy.io import loadmat


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


# Example usage (if data is prepared):
# mu_P, cov_P = sample_price_day(Pt_day=np.random.rand(24*7), t=1, Season='Summer')
