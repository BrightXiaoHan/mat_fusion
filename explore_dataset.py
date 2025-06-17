import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file (HDF5 format)
mat_file = 'assets/sea_ice_dataset_withoutint.mat'
data = h5py.File(mat_file, 'r')

# Explore the structure
print("Keys in the .mat file:")
for key in data.keys():
    print(f"  {key}: {type(data[key])}, shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")

# Display basic info about each variable
print("\nDetailed information:")
for key in data.keys():
    var = data[key]
    print(f"\n{key}:")
    print(f"  Type: {type(var)}")
    if hasattr(var, 'shape'):
        print(f"  Shape: {var.shape}")
        print(f"  Data type: {var.dtype}")
        if var.size > 0:
            arr = np.array(var)
            print(f"  Min: {np.min(arr)}")
            print(f"  Max: {np.max(arr)}")
            print(f"  Mean: {np.mean(arr)}")

data.close()