import torch
import torch.nn.functional as F
import os
import sys # Add sys import

# --- Load Data ---
# Construct the path relative to this script's location
script_dir = os.path.dirname(__file__) # Directory of the current script
project_root = os.path.dirname(script_dir) # Go up one level to the project root
data_dir = os.path.join(project_root, 'data') # Path to the data directory
src_dir = os.path.join(project_root, 'src') # Path to the src directory

# Add src directory to Python path to allow importing support_functions
sys.path.append(src_dir)
from support_functions import empirical_cross_covariance_with_torch_cov

x_load_path = os.path.join(data_dir, 'x_data.pt')
z_load_path = os.path.join(data_dir, 'z_labels.pt')

# Check if data files exist before loading
if not os.path.exists(x_load_path) or not os.path.exists(z_load_path):
    print(f"Error: Data files not found.")
    print(f"Please run data/data_generation.py first to generate {x_load_path} and {z_load_path}")
    exit()

x = torch.load(x_load_path)
z = torch.load(z_load_path)

print(f"Loaded data tensor 'x' from {x_load_path} with shape: {x.shape}")
print(f"Loaded label tensor 'z' from {z_load_path} with shape: {z.shape}")

# Determine num_classes from the loaded labels
num_classes = int(z.max()) + 1 # Assumes labels are 0-indexed
print(f"Inferred number of classes: {num_classes}")

# --- One-Hot Encode Labels ---
# Ensure z is long type and convert to float for calculations
z_one_hot = F.one_hot(z, num_classes=num_classes).float()
print(f"Shape of one-hot label tensor 'z_one_hot': {z_one_hot.shape}")

# --- Calculate Cov(x, x) ---
# Use the imported function
cov_xx = empirical_cross_covariance_with_torch_cov(x, x, correction=1)

# --- Calculate Cov(x, z_one_hot) ---
# Use the imported function
cov_xz = empirical_cross_covariance_with_torch_cov(x, z_one_hot, correction=1)

# --- Print Results ---
print("\n--- Covariance Matrices ---")
print(f"Cov(x, x) (Shape: {cov_xx.shape}):")
print(cov_xx)
print()
print(f"Cov(x, z_one_hot) (Shape: {cov_xz.shape}):")
print(cov_xz)
