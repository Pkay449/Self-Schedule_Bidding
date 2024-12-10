# %%
import matlab.engine
import numpy as np

# %%

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define sample inputs
phi = matlab.double([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # A 3x3 matrix

Y = matlab.double([3, 5, 7])  # Target state (row vector)

weights_lsqlin = matlab.double([1, 0.5, 0.2])  # Feature scaling weights

# Run the MATLAB function
weights = eng.VRx_weights(phi, Y, weights_lsqlin)

# Display the result
print("Computed weights:", weights)

# Close MATLAB engine
eng.quit()
# %%

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the input variable T
T = 5  # Adjust this value as needed

# Call the MATLAB function badp_weights
weights = eng.badp_weights(T)
# save array to '../../notebooks/matlab_badpWeights.npy'
np.save("../../notebooks/matlab_badpWeights.npy", weights)
print("Weights saved successfully!")

# Close MATLAB engine
eng.quit()
# %%

# Start MATLAB engine
eng = matlab.engine.start_matlab()

eng.badp_w()

# Close MATLAB engine
eng.quit()

# %%

# Start MATLAB engine
import matlab.engine

eng = matlab.engine.start_matlab()

# Define input arguments
Pt_day = matlab.double([50, 55, 60, 65, 70, 75, 80, 85])  # Example row vector
t = 1  # Current stage
Season = "Summer"  # Specify the season

# Call the MATLAB function
mu_P, cov_P = eng.sample_price_day(Pt_day, t, Season, nargout=2)

print("Mean prices (mu_P):", mu_P)
print("Covariance matrix (cov_P):", cov_P)


# Close MATLAB engine
eng.quit()


# %%
