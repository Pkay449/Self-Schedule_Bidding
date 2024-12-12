# %%
import matlab.engine
import numpy as np

# %%

# %%

# Start MATLAB engine
eng = matlab.engine.start_matlab()

eng.tmp()

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
