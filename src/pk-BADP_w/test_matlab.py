#%%
import matlab.engine
import numpy as np

#%%

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define sample inputs
phi = matlab.double([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])  # A 3x3 matrix

Y = matlab.double([3, 5, 7])  # Target state (row vector)

weights_lsqlin = matlab.double([1, 0.5, 0.2])  # Feature scaling weights

# Run the MATLAB function
weights = eng.VRx_weights(phi, Y, weights_lsqlin)

# Display the result
print("Computed weights:", weights)

# Close MATLAB engine
eng.quit()
#%%


# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the input variable T
T = 5  # Adjust this value as needed

# Call the MATLAB function badp_weights
weights = eng.badp_weights(T)
# save array to '../../notebooks/matlab_badpWeights.npy'
np.save('../../notebooks/matlab_badpWeights.npy', weights)
print("Weights saved successfully!")

# Close MATLAB engine
eng.quit()
#%%


# Start MATLAB engine
eng = matlab.engine.start_matlab()

eng.badp_w()

# %%
