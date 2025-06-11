import torch
import matplotlib.pyplot as plt
import numpy as np # Often needed alongside PyTorch, though not strictly for generation here
import os # To handle file paths and directories

# --- Parameters ---
n_samples_per_cluster = 400  # Number of points in each cluster
random_seed = 42           # for reproducibility
torch.manual_seed(random_seed) # Set seed for PyTorch random number generation

# --- Define Cluster Properties ---

# Cluster 0 (Orange - Label 0)
# Centered slightly right and down, elongated from top-left to bottom-right
mean0 = torch.tensor([1.5, -1.0])
# Covariance matrix: Controls spread and rotation.
# Larger variance on diagonal = more spread.
# Off-diagonal values control correlation/rotation. Negative correlation here.
cov0 = torch.tensor([[2.5, -1.0],
                     [-1.0, 1.0]]) # Increased variance along one diagonal

# Cluster 1 (Blue - Label 1)
# Centered slightly left and up, elongated from bottom-left to top-right
mean1 = torch.tensor([-1.5, 1.0])
# Positive correlation for this cluster's elongation
cov1 = torch.tensor([[2.5, 1.0],
                     [1.0, 1.0]]) # Increased variance along the other diagonal


# --- Generate Data ---

# Create multivariate normal distributions
dist0 = torch.distributions.MultivariateNormal(mean0, covariance_matrix=cov0)
dist1 = torch.distributions.MultivariateNormal(mean1, covariance_matrix=cov1)

# Sample points from each distribution
points0 = dist0.sample((n_samples_per_cluster,))
points1 = dist1.sample((n_samples_per_cluster,))

# Create corresponding labels
# Using torch.long is common for labels in classification tasks
labels0 = torch.zeros(n_samples_per_cluster, dtype=torch.long) # Label 0 for the first cluster (orange)
labels1 = torch.ones(n_samples_per_cluster, dtype=torch.long)  # Label 1 for the second cluster (blue)

# Combine points and labels
# Concatenate along dimension 0 (rows)
x = torch.cat([points0, points1], dim=0)
z = torch.cat([labels0, labels1], dim=0) # Variable 'z' for labels as requested

# --- Shuffle the Dataset ---
# It's crucial to shuffle so that samples aren't ordered by class
total_samples = x.shape[0]
shuffle_indices = torch.randperm(total_samples) # Generate random permutation of indices

x = x[shuffle_indices]
z = z[shuffle_indices]

# --- Verify Output ---
print(f"Shape of data tensor 'x': {x.shape}")
print(f"Shape of label tensor 'z': {z.shape}")
print(f"Data type of 'x': {x.dtype}")
print(f"Data type of 'z': {z.dtype}")

print("\nFirst 5 data points (x):")
print(x[:5])
print("\nFirst 5 labels (z):")
print(z[:5])


# --- Save the Data --- 
data_dir = os.path.dirname(__file__) # Get the directory where the script is located
os.makedirs(data_dir, exist_ok=True) # Create the data directory if it doesn't exist

x_save_path = os.path.join(data_dir, 'x_data.pt')
z_save_path = os.path.join(data_dir, 'z_labels.pt')

torch.save(x, x_save_path)
torch.save(z, z_save_path)

print(f"\nData saved to {x_save_path}")
print(f"Labels saved to {z_save_path}")

# --- Visualize the Generated Data (Optional but Recommended) ---
plt.style.use('seaborn-v0_8-darkgrid') # Use a style similar to the screenshot
plt.figure(figsize=(7, 7))

# Plot points for label 0
plt.scatter(x[z == 0, 0].numpy(), x[z == 0, 1].numpy(),
            c='orange', label='Label 0', alpha=0.5, edgecolors='k', linewidth=0.5)

# Plot points for label 1
plt.scatter(x[z == 1, 0].numpy(), x[z == 1, 1].numpy(),
            c='blue', label='Label 1', alpha=0.5, edgecolors='k', linewidth=0.5)

plt.title("Generated PyTorch Dataset", fontsize=16, fontweight='bold')
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Set axis limits similar to the screenshot for better comparison
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.gca().set_aspect('equal', adjustable='box') # Ensure x and y axes have the same scale visually
plt.tick_params(axis='both', which='major', labelsize=10)

plt.show()