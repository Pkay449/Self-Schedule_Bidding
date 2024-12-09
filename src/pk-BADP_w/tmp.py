import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Add the path where VRx_weights.m is located
eng.addpath(r'/path/to/directory', nargout=0)

# Define inputs
phi = matlab.double([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
Y = matlab.double([3, 5, 7])
weights_lsqlin = matlab.double([1, 0.5, 0.2])

# Call the function
weights = eng.VRx_weights(phi, Y, weights_lsqlin)

# Print results
print("Computed weights:", weights)

# Stop MATLAB engine
eng.quit()

