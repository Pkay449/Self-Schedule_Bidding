# %%
import matlab.engine
import numpy as np
import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
# set matlab path as current path
eng = matlab.engine.start_matlab()
eng.addpath(current_path, nargout=0)
eng.quit()



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
