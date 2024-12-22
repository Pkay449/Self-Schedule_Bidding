# %%
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import h5py

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load src/pk-BADP_w/Data/P_day_ahead_test_all.mat
data = loadmat("Data/P_day_ahead_test_all.mat")
data_DA = data['P_day_0'].ravel()

# Load src/pk-BADP_w/Data/P_intraday_test_all.mat
data = loadmat("Data/P_intraday_test_all.mat")
data_ID = data['P_intraday_0'].ravel()
data_ID.shape

# Reshape DA by days
data_DA = data_DA.reshape(-1, 24)
data_ID = data_ID.reshape(-1, 96)

# Create a dataset where for each row, we have a day of DA and ID
# and the target is the next day DA and ID

rows = []
for i in range(1, len(data_DA)):
    rows.append({
        'DA_day_t': data_DA[i-1],
        'ID_day_t': data_ID[i-1],
        'DA_day_t+1': data_DA[i],
        'ID_day_t+1': data_ID[i]
    })

# Convert the list of rows to a DataFrame
test_df = pd.DataFrame(rows)

test_df
# %%

# Save the DataFrame to HDF5
with h5py.File("Results/test_dataset.h5", "w") as h5file:
    for col in test_df.columns:
        h5file.create_dataset(col, data=np.array(test_df[col].tolist()))


# %%
# Load the DataFrame from HDF5
with h5py.File("Results/test_dataset.h5", "r") as h5file:
    loaded_data = {
        col: np.array(h5file[col])
        for col in h5file.keys()
    }

# Reconstruct as a numpy structured array instead of a DataFrame
loaded_np_array = {col: loaded_data[col] for col in loaded_data}

# Access as numpy array
DA_day_t_array = loaded_np_array['DA_day_t']
print(type(DA_day_t_array))  # Output: <class 'numpy.ndarray'>
print(DA_day_t_array.shape)  # Shape of the NumPy array

# %%
DA_day_t_array
# %%
DA_day_t_array[:7]
# %%
DA_day_t_array[:7].ravel()
# %%
