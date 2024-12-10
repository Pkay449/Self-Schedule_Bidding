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

sample_price_day_inputs_path = '../BADP_w/debug_sample_price_day.mat'
sample_price_intraday_inputs_path = '../BADP_w/debug_sample_price_intraday.mat'

# test sample_price_day

# Start MATLAB engine
eng = matlab.engine.start_matlab()
inputs_day = eng.load(sample_price_day_inputs_path)
P_day = inputs_day['P_day']
t = inputs_day['t']
Season = inputs_day['Season']
eng.sample_price_day(P_day,t,Season)




# %%
