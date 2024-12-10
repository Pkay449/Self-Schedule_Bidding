import numpy as np
from scipy.io import loadmat
import os

def badp_weights(T=5):
    D = 7    # days in forecast
    Season = 'Summer'
    gamma = 1.0  # discount factor for futures prices

    # Load data
    # Assuming the .mat files have variables 'beta_day_ahead' and 'beta_intraday'
    beta_day_ahead_path = os.path.join('Data', f'beta_day_ahead_{Season}.mat')
    beta_intraday_path = os.path.join('Data', f'beta_intraday_{Season}.mat')
    
    data_day_ahead = loadmat(beta_day_ahead_path)
    data_intraday = loadmat(beta_intraday_path)

    # Extract arrays from the loaded .mat structures
    # Adjust the keys if needed, depending on the exact variable names inside the mat files.
    beta_day_ahead = data_day_ahead['beta_day_ahead']
    beta_intraday = data_intraday['beta_intraday']

    # Remove the first 8 columns
    beta_day_ahead = beta_day_ahead[:, 8:]
    beta_intraday = beta_intraday[:, 8:]

    # Initialize influence matrices
    einfluss_day = np.zeros((T, 24 * D))
    einfluss_intraday = np.zeros((T, 24 * 4 * D))

    # Compute day-ahead weights
    for t in range(T):
        for i in range(24 * D):
            Pt_day = np.zeros((1, 24 * D))
            Pt_day[0, i] = 1
            Pt_intraday = np.zeros((1, 96 * D))

            Wt_day_mat = np.zeros((T, 24))
            Wt_intraday_mat = np.zeros((T, 96))

            for t_strich in range(t, T):
                # Wt_day = Pt_day * beta_day_ahead'
                # beta_day_ahead is (24 x 168), so beta_day_ahead.T is (168 x 24)
                Wt_day = Pt_day @ beta_day_ahead.T  # 1x24

                # Wt_intraday = [Wt_day, Pt_day, Pt_intraday]*beta_intraday'
                # Create the combined array [Wt_day, Pt_day, Pt_intraday]
                combined = np.hstack([Wt_day, Pt_day, Pt_intraday])  # 1 x (24+168+672=864)
                Wt_intraday = combined @ beta_intraday.T  # 1x96

                # Update Pt_day and Pt_intraday for next iteration
                # Pt_day = [Wt_day, Pt_day(1:end-24)] in MATLAB
                # We must shift Pt_day by 24 steps and prepend Wt_day
                Pt_day = np.hstack([Wt_day, Pt_day[:, :-24]])  # shape stays 1x168

                # Pt_intraday = [Wt_intraday, Pt_intraday(1:end-96)]
                Pt_intraday = np.hstack([Wt_intraday, Pt_intraday[:, :-96]])  # shape stays 1x672

                Wt_day_mat[t_strich, :] = Wt_day
                Wt_intraday_mat[t_strich, :] = Wt_intraday

                einfluss_day[t, i] += (4 * np.sum(np.abs(Wt_day)) + np.sum(np.abs(Wt_intraday))) * (gamma ** (t_strich - t))

    # Compute intraday weights
    for t in range(T):
        for i in range(24 * 4 * D):
            Pt_day = np.zeros((1, 24 * D))
            Pt_intraday = np.zeros((1, 96 * D))
            Pt_intraday[0, i] = 1

            Wt_day_mat = np.zeros((T, 24))
            Wt_intraday_mat = np.zeros((T, 96))

            for t_strich in range(t, T):
                Wt_day = Pt_day @ beta_day_ahead.T  # 1x24
                combined = np.hstack([Wt_day, Pt_day, Pt_intraday])  # 1x864
                Wt_intraday = combined @ beta_intraday.T  # 1x96

                Pt_day = np.hstack([Wt_day, Pt_day[:, :-24]])
                Pt_intraday = np.hstack([Wt_intraday, Pt_intraday[:, :-96]])

                Wt_day_mat[t_strich, :] = Wt_day
                Wt_intraday_mat[t_strich, :] = Wt_intraday

                einfluss_intraday[t, i] += (4 * np.sum(np.abs(Wt_day)) + np.sum(np.abs(Wt_intraday))) * (gamma ** (t_strich - t))

    weights = np.hstack([einfluss_day, einfluss_intraday])

    return weights

# Example usage:
# weights = badp_weights()
# If you want to save:
# from scipy.io import savemat
# savemat('weights.mat', {'weights': weights})


if __name__ == "__main__":
    import sys
    import os
    
    # set current path as working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    weights = badp_weights(T=5)
    # save array to '../../notebooks/udays_badpWeights.npy'
    np.save('../../notebooks/pk_badpWeights.npy', weights)
    print("Weights saved successfully!")