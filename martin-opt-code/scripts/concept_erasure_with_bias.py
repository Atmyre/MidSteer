"""
Script to solve the optimization problem for an *affine* transformation P:

min_P E_x[ || Py - x ||_M^2 ] subject to Cov(Py, z) = 0

where y = [1, x] is the augmented feature vector.
The problem is solved using the Condat-Vu primal-dual algorithm.

Algorithm Steps:
1. Load data x (features) and z (labels) from the 'data/' directory.
2. Create augmented data y by prepending a column of ones to x.
3. Convert labels z into one-hot encoded vectors.
4. Compute covariance matrices Sigma_yy = Cov(y, y), Sigma_yz = Cov(y, z_one_hot),
   and cross-covariance Sigma_xy = Cov(x, y) using the function from
   'src/support_functions.py'.
5. Initialise the metric M (default: identity matrix, dimension n x n).
6. Initialise the optimization variables P (e.g., [0, I], dimension n x (n+1)) and
   Lambda (dual variable, e.g., zero matrix, dimension n x num_classes).
7. Calculate operator norms ||M||, ||Sigma_yy||, ||Sigma_yz||.
8. Determine appropriate step sizes t (primal) and s (dual) based on
   Condat-Vu stability conditions:
   * 0 < t < 1 / (||M|| * ||Sigma_yy|| + s * ||Sigma_yz||^2)
    * 0 < s
9. Iterate using the Condat-Vu update rules:
   P_k = P_{k-1} - t * ( M @ (P_{k-1} @ Sigma_yy - Sigma_xy) + Lambda_{k-1} @ Sigma_yz.T )
   Lambda_k = Lambda_{k-1} + s * ( (2*P_k - P_{k-1}) @ Sigma_yz )
10. Check the stopping criterion based on the average norm of the
    optimality conditions (residuals):
    - Primal residual: M @ (P @ Sigma_yy - Sigma_xy) + Lambda @ Sigma_yz.T
    - Dual residual: P @ Sigma_yz
    Stop when 0.5 * (||primal_res|| + ||dual_res||) < TOLERANCE.
11. Print the final matrix P, residual norms, and iteration count.
12. Visualise the effect of the affine transformation Py on the data.
"""
import torch
import torch.nn.functional as F
import os
import sys
import time
import matplotlib.pyplot as plt # Add matplotlib import

# --- Configuration ---
MAX_ITER = 10000
TOLERANCE = 1e-6
STEP_SIZE_FACTOR = 0.9 # Factor to ensure strict inequality for step sizes

# --- Setup Paths ---
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'data')
src_dir = os.path.join(project_root, 'src')

# Add src directory to Python path
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from support_functions import empirical_cross_covariance_with_torch_cov
except ImportError:
    print(f"Error: Could not import 'empirical_cross_covariance_with_torch_cov' from {src_dir}.")
    print("Please ensure 'support_functions.py' exists in the 'src' directory.")
    exit()

# --- Load Data ---
x_load_path = os.path.join(data_dir, 'x_data.pt')
z_load_path = os.path.join(data_dir, 'z_labels.pt')

if not os.path.exists(x_load_path) or not os.path.exists(z_load_path):
    print(f"Error: Data files not found.")
    print(f"Please run data/data_generation.py first to generate {x_load_path} and {z_load_path}")
    exit()

try:
    x = torch.load(x_load_path)
    z = torch.load(z_load_path)
    print(f"Loaded data tensor 'x' from {x_load_path} with shape: {x.shape}")
    print(f"Loaded label tensor 'z' from {z_load_path} with shape: {z.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Preprocess Data ---
n = x.shape[1] # Dimension of x
N = x.shape[0] # Number of samples

# Create augmented data y = [1, x]
ones = torch.ones(N, 1, dtype=x.dtype, device=x.device)
y = torch.cat((ones, x), dim=1)
n_aug = y.shape[1] # Dimension of y (n+1)
print(f"Created augmented data tensor 'y' with shape: {y.shape}")

if z.dim() == 0: # Handle case where z might be a single scalar tensor
    z = z.unsqueeze(0)
if z.max().item() < 0:
     print(f"Error: Invalid class labels found (max label < 0).")
     exit()

num_classes = int(z.max().item()) + 1 # Assumes labels are 0-indexed
print(f"Inferred number of classes: {num_classes}")

try:
    # Ensure z is long type for one_hot
    z_one_hot = F.one_hot(z.long(), num_classes=num_classes).float()
    print(f"Shape of one-hot label tensor 'z_one_hot': {z_one_hot.shape}")
except Exception as e:
    print(f"Error during one-hot encoding: {e}")
    exit()

# --- Compute Covariances ---
try:
    # Ensure data is float for covariance calculation
    x = x.float()
    y = y.float() # Ensure y is float
    z_one_hot = z_one_hot.float()

    # Calculate Cov(y, y)
    sigma_yy = empirical_cross_covariance_with_torch_cov(y, y, correction=1)
    print(f"Computed Cov(y, y) (Sigma_yy) with shape: {sigma_yy.shape}")

    # Calculate Cov(y, z_one_hot)
    sigma_yz = empirical_cross_covariance_with_torch_cov(y, z_one_hot, correction=1)
    print(f"Computed Cov(y, z_one_hot) (Sigma_yz) with shape: {sigma_yz.shape}")

    # Calculate Cov(x, y) - needed for the objective function gradient
    sigma_xy = empirical_cross_covariance_with_torch_cov(x, y, correction=1)
    print(f"Computed Cov(x, y) (Sigma_xy) with shape: {sigma_xy.shape}")

except ValueError as e:
    print(f"Error computing covariances: {e}")
    print("This might happen if there are too few samples (N < 2 for unbiased estimate).")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during covariance calculation: {e}")
    exit()

# --- Initialise Algorithm Variables ---
M = torch.eye(n, dtype=x.dtype, device=x.device) # M = Identity (n x n)
# Initial P = [0, I] so that Py = x initially
P = torch.zeros(n, n_aug, dtype=x.dtype, device=x.device)
P[:, 1:] = torch.eye(n, dtype=x.dtype, device=x.device)
print(f"Initialised P with shape: {P.shape}")

# Lambda should have dimensions n x num_classes
Lambda = torch.zeros(n, num_classes, dtype=x.dtype, device=x.device)
print(f"Initialised Lambda with shape: {Lambda.shape}")

# --- Calculate Operator Norms and Step Sizes ---
try:
    norm_M = torch.linalg.norm(M, ord=2).item() if n > 0 else 1.0
    # Use norms of the new covariance matrices
    norm_sigma_yy = torch.linalg.norm(sigma_yy, ord=2).item() if n_aug > 0 else 0.0
    norm_sigma_yz = torch.linalg.norm(sigma_yz, ord=2).item() if sigma_yz.numel() > 0 else 0.0

    print(f"\n--- Operator Norms ---")
    print(f"||M|| = {norm_M:.4f}")
    print(f"||Sigma_yy|| = {norm_sigma_yy:.4f}") # Changed from sigma_xx
    print(f"||Sigma_yz|| = {norm_sigma_yz:.4f}") # Changed from sigma_xz

    # --- Step Size Calculation (User Specified Method) ---
    # Fix s to a specific value
    s = 1.0
    print(f"\n--- Step Sizes ---")
    print(f"Fixed s = {s:.4e}")

    # Calculate t based on the condition: t < 1 / (||M||*||Sigma_yy|| + s*||Sigma_yz||^2)
    denominator = norm_M * norm_sigma_yy + s * norm_sigma_yz**2 # Use yy and yz norms

    if denominator <= 1e-9:
        print("Warning: Denominator for t calculation is close to zero. Cannot determine step size t reliably.")
        # Set a default small step size or handle as error
        t = 1e-4
        t_upper_bound = float('inf') # Indicate the bound couldn't be calculated properly
    else:
        t_upper_bound = 1.0 / denominator
        t = STEP_SIZE_FACTOR * t_upper_bound

    print(f"Chosen t = {t:.4e} (Upper bound based on fixed s: {t_upper_bound:.4e})")

except Exception as e:
    print(f"Error calculating norms or step sizes: {e}")
    exit()

# --- Condat-Vu Iteration ---
print("\n--- Starting Condat-Vu Algorithm ---")
start_time = time.time()
for k in range(MAX_ITER):
    P_prev = P.clone()
    Lambda_prev = Lambda.clone()

    # Update P using the gradient for ||Py - x||^2 and the constraint Cov(Py, z) = 0
    grad_P = M @ (P_prev @ sigma_yy - sigma_xy) + Lambda_prev @ sigma_yz.T
    P = P_prev - t * grad_P

    # Update Lambda
    P_extrapolated = 2 * P - P_prev
    grad_Lambda = P_extrapolated @ sigma_yz
    Lambda = Lambda_prev + s * grad_Lambda

    # Check stopping criterion (Optimality conditions)
    # Primal residual: M @ (P @ Sigma_yy - Sigma_xy) + Lambda @ Sigma_yz.T
    primal_residual = M @ (P @ sigma_yy - sigma_xy) + Lambda @ sigma_yz.T
    # Dual residual: P @ Sigma_yz
    dual_residual = P @ sigma_yz

    norm_primal_res = torch.linalg.norm(primal_residual).item()
    norm_dual_res = torch.linalg.norm(dual_residual).item()
    avg_residual_norm = 0.5 * (norm_primal_res + norm_dual_res)

    if (k + 1) % 10 == 0:
        print(f"Iter {k+1}/{MAX_ITER}, Avg Residual Norm: {avg_residual_norm:.4e}")

    if avg_residual_norm < TOLERANCE:
        print(f"\nConverged after {k+1} iterations.")
        break
else: # Runs if the loop completes without break
    print(f"\nWarning: Algorithm did not converge within {MAX_ITER} iterations.")
    print(f"Final Avg Residual Norm: {avg_residual_norm:.4e}")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

# --- Output Results ---
print("\n--- Results ---")
print(f"Final Matrix P (Shape: {P.shape}):")
# print(P) # Optionally print the full matrix
print(f"Final Lambda (Shape: {Lambda.shape}):")
# print(Lambda) # Optionally print the full matrix
print(f"Final Primal Residual Norm: {norm_primal_res:.4e}")
print(f"Final Dual Residual Norm (Constraint Violation Norm): {norm_dual_res:.4e}")
print(f"Final Average Residual Norm: {avg_residual_norm:.4e}")

# --- Optional: Save the result ---
output_dir = os.path.join(project_root, 'results')
os.makedirs(output_dir, exist_ok=True)
P_save_path = os.path.join(output_dir, 'P_optimal_with_bias.pt')
torch.save(P, P_save_path)
print(f"\nSaved optimal matrix P to {P_save_path}")

# --- Visualise the Transformation ---
print("\n--- Visualising Transformation ---")

# Load the original data (already loaded as x, z)
# Load the computed P matrix (already in memory as P)
# Use the augmented data y

# Apply the transformation P to y: x_transformed = Py
# Ensure P is on the same device and dtype as y if necessary
P = P.to(y.device, dtype=y.dtype)
x_transformed = y @ P.T # Apply transformation: x_new = Py

# Convert to numpy for plotting
x_np = x.cpu().numpy() # Original x for plotting
x_transformed_np = x_transformed.cpu().detach().numpy() # Detach if P requires grad
z_np = z.cpu().numpy()

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Plot Original Data
axes[0].scatter(x_np[z_np == 0, 0], x_np[z_np == 0, 1],
                c='orange', label='Label 0', alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0].scatter(x_np[z_np == 1, 0], x_np[z_np == 1, 1],
                c='blue', label='Label 1', alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0].set_title("Original Data (x)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Feature 1", fontsize=12)
axes[0].set_ylabel("Feature 2", fontsize=12)
axes[0].legend()
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axes[0].set_aspect('equal', adjustable='box')
axes[0].tick_params(axis='both', which='major', labelsize=10)
# Optional: Set consistent limits if needed
x_lim = (min(x_np[:, 0].min(), x_transformed_np[:, 0].min()) - 1,
         max(x_np[:, 0].max(), x_transformed_np[:, 0].max()) + 1)
y_lim = (min(x_np[:, 1].min(), x_transformed_np[:, 1].min()) - 1,
         max(x_np[:, 1].max(), x_transformed_np[:, 1].max()) + 1)
axes[0].set_xlim(x_lim)
axes[0].set_ylim(y_lim)

# Plot Transformed Data
axes[1].scatter(x_transformed_np[z_np == 1, 0], x_transformed_np[z_np == 1, 1],
                c='blue', label='Label 1', alpha=0.5, edgecolors='k', linewidth=0.5)
axes[1].scatter(x_transformed_np[z_np == 0, 0], x_transformed_np[z_np == 0, 1],
                c='orange', label='Label 0', alpha=0.5, edgecolors='k', linewidth=0.5)
axes[1].set_title("Transformed Data (Py)", fontsize=14, fontweight='bold') # Updated title
axes[1].set_xlabel("Transformed Feature 1", fontsize=12)
axes[1].set_ylabel("Transformed Feature 2", fontsize=12)
axes[1].legend()
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axes[1].set_aspect('equal', adjustable='box')
axes[1].tick_params(axis='both', which='major', labelsize=10)
axes[1].set_xlim(x_lim)
axes[1].set_ylim(y_lim)

plt.suptitle("Effect of Affine Transformation P satisfying Cov(Py, z) = 0", fontsize=16, fontweight='bold') # Updated suptitle
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

# Save the plot
plot_save_path = os.path.join(output_dir, 'transformation_visualisation_with_bias.png')
plt.savefig(plot_save_path)
print(f"Saved visualisation to {plot_save_path}")

plt.show()
