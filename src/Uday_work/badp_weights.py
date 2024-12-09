import numpy as np
from scipy.io import loadmat

def badp_weights(T=5):
    # Parameters
    D = 7    # days in forecast
    Season = 'Summer'
    gamma = 1.0  # discount factor for futures prices

    # Load data
    # Adjust file paths and variable keys according to your environment and data files
    beta_day_ahead_data = loadmat(f'Data/beta_day_ahead_{Season}.mat')
    beta_intraday_data = loadmat(f'Data/beta_intraday_{Season}.mat')

    beta_day_ahead = beta_day_ahead_data['beta_day_ahead']
    beta_intraday = beta_intraday_data['beta_intraday']

    # In MATLAB:
    # beta_day_ahead(:,1:8)=[]; and beta_intraday(:,1:8)=[] 
    # means we remove the first 8 columns of beta_day_ahead and beta_intraday
    beta_day_ahead = beta_day_ahead[:, 8:]
    beta_intraday = beta_intraday[:, 8:]

    # Initialize arrays
    # einfluss_day has shape (T, 24*D)
    einfluss_day = np.zeros((T, 24*D))
    # einfluss_intraday has shape (T, 24*4*D) = (T, 96*D)
    einfluss_intraday = np.zeros((T, 96*D))

    # Compute einfluss_day
    for t in range(T):
        for i in range(24*D):
            Pt_day = np.zeros(24*D)
            Pt_day[i] = 1
            Pt_intraday = np.zeros(96*D)
            # Wt_day_mat and Wt_intraday_mat are used but not saved; can be omitted or kept for clarity
            Wt_day_mat = np.zeros((T, 24))
            Wt_intraday_mat = np.zeros((T, 96))

            for t_strich in range(t, T):
                # Wt_day = Pt_day * beta_day_ahead'
                # In MATLAB: Wt_day is 1x(24*D)* ( (24*D)x24 ) = 1x24
                Wt_day = Pt_day @ beta_day_ahead.T  # shape: (24,)
                
                # Wt_intraday = [Wt_day, Pt_day, Pt_intraday]*beta_intraday'
                # Concatenate Wt_day (1x24), Pt_day (1x(24*D)), and Pt_intraday (1x(96*D))
                # Then multiply by beta_intraday'
                combined_intraday = np.concatenate([Wt_day, Pt_day, Pt_intraday])
                Wt_intraday = combined_intraday @ beta_intraday.T  # shape: (96,)

                # Update Pt_day and Pt_intraday as per code:
                Pt_day = np.concatenate([Wt_day, Pt_day[:-24]])
                Pt_intraday = np.concatenate([Wt_intraday, Pt_intraday[:-96]])

                Wt_day_mat[t_strich, :] = Wt_day
                Wt_intraday_mat[t_strich, :] = Wt_intraday

                einfluss_day[t, i] += (4 * np.sum(np.abs(Wt_day)) + np.sum(np.abs(Wt_intraday))) * (gamma ** (t_strich - t))

    # Compute einfluss_intraday
    for t in range(T):
        for i in range(96*D):
            Pt_day = np.zeros(24*D)
            Pt_intraday = np.zeros(96*D)
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

                einfluss_intraday[t, i] += (4 * np.sum(np.abs(Wt_day)) + np.sum(np.abs(Wt_intraday))) * (gamma ** (t_strich - t))

    weights = np.hstack([einfluss_day, einfluss_intraday])
    return weights

# Example usage (if data and functions are ready):
# weights = badp_weights(T=5)

if __name__ == "__main__":
    import sys
    import os
    
    # set current path as working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    weights = badp_weights(T=5)
    # save array to '../../notebooks/udays_badpWeights.npy'
    np.save('../../notebooks/udays_badpWeights.npy', weights)
    print("Weights saved successfully!")