"""
Script to solve the optimization problem:

min_P E_x[ || Px - x ||_M^2 ] subject to Cov(Px, z) = 0

using the Condat-Vu primal-dual algorithm.

Algorithm Steps:
1. Load data x (features) and z (labels) from the 'data/' directory.
2. Convert labels z into one-hot encoded vectors.
3. Compute covariance matrix Sigma_xx = Cov(x, x) and cross-covariance
   matrix Sigma_xz = Cov(x, z_one_hot) using the function from
   'src/support_functions.py'.
4. Initialise the metric M (default: identity matrix).
5. Initialise the optimization variables P (e.g., identity matrix) and
   Lambda (dual variable, e.g., zero matrix).
6. Calculate operator norms ||M||, ||Sigma_xx||, ||Sigma_xz||.
7. Determine appropriate step sizes t (primal) and s (dual) based on
   Condat-Vu stability conditions:
   * 0 < t < 1 / (||M|| * ||Sigma_xx|| + s * ||Sigma_xz||^2)
    * 0 < s
8. Iterate using the Condat-Vu update rules:
   P_k = P_{k-1} - t * ( M @ (P_{k-1} - I) @ Sigma_xx + Lambda_{k-1} @ Sigma_xz.T )
   Lambda_k = Lambda_{k-1} + s * ( (2*P_k - P_{k-1}) @ Sigma_xz )
9. Check the stopping criterion based on the average norm of the
   optimality conditions (residuals):
   - Primal residual: M @ (P - I) @ Sigma_xx + Lambda @ Sigma_xz.T
   - Dual residual: P @ Sigma_xz
   Stop when 0.5 * (||primal_res|| + ||dual_res||) < TOLERANCE.
10. Print the final matrix P, residual norms, and iteration count.
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
    z_one_hot = z_one_hot.float()

    # Calculate Cov(x, x)
    sigma_xx = empirical_cross_covariance_with_torch_cov(x, x, correction=1)
    print(f"Computed Cov(x, x) (Sigma_xx) with shape: {sigma_xx.shape}")

    # Calculate Cov(x, z_one_hot)
    sigma_xz = empirical_cross_covariance_with_torch_cov(x, z_one_hot, correction=1)
    print(f"Computed Cov(x, z_one_hot) (Sigma_xz) with shape: {sigma_xz.shape}")

except ValueError as e:
    print(f"Error computing covariances: {e}")
    print("This might happen if there are too few samples (N < 2 for unbiased estimate).")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during covariance calculation: {e}")
    exit()

# --- Initialise Algorithm Variables ---
M = torch.eye(n, dtype=x.dtype, device=x.device) # M = Identity
P = torch.eye(n, dtype=x.dtype, device=x.device) # Initial P = Identity
Lambda = torch.zeros_like(sigma_xz, dtype=x.dtype, device=x.device) # Initial Lambda = 0

# --- Calculate Operator Norms and Step Sizes ---
try:
    norm_M = torch.linalg.norm(M, ord=2).item() if n > 0 else 1.0
    norm_sigma_xx = torch.linalg.norm(sigma_xx, ord=2).item() if n > 0 else 0.0
    norm_sigma_xz = torch.linalg.norm(sigma_xz, ord=2).item() if sigma_xz.numel() > 0 else 0.0

    print(f"\\n--- Operator Norms ---")
    print(f"||M|| = {norm_M:.4f}")
    print(f"||Sigma_xx|| = {norm_sigma_xx:.4f}")
    print(f"||Sigma_xz|| = {norm_sigma_xz:.4f}")

    # --- Step Size Calculation (User Specified Method) ---
    # Fix s to a specific value
    s = 1.0
    print(f"\\n--- Step Sizes ---")
    print(f"Fixed s = {s:.4e}")

    # Calculate t based on the condition: t < 1 / (||M||*||Sigma_xx|| + s*||Sigma_xz||^2)
    denominator = norm_M * norm_sigma_xx + s * norm_sigma_xz**2

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

    # Update P
    grad_P = M @ (P_prev - torch.eye(n, dtype=P.dtype, device=P.device)) @ sigma_xx + Lambda_prev @ sigma_xz.T
    P = P_prev - t * grad_P

    # Update Lambda
    P_extrapolated = 2 * P - P_prev
    grad_Lambda = P_extrapolated @ sigma_xz
    Lambda = Lambda_prev + s * grad_Lambda

    # Check stopping criterion (Optimality conditions)
    primal_residual = M @ (P - torch.eye(n, dtype=P.dtype, device=P.device)) @ sigma_xx + Lambda @ sigma_xz.T
    dual_residual = P @ sigma_xz

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
P_save_path = os.path.join(output_dir, 'P_optimal.pt')
torch.save(P, P_save_path)
print(f"\\nSaved optimal matrix P to {P_save_path}")

# --- Visualise the Transformation ---
print("\\n--- Visualising Transformation ---")

# Load the original data (already loaded as x, z)
# Load the computed P matrix (already in memory as P)

# Apply the transformation P to x
# Ensure P is on the same device and dtype as x if necessary
P = P.to(x.device, dtype=x.dtype)
x_transformed = x @ P.T # Apply transformation: x_new = Px

# Convert to numpy for plotting
x_np = x.cpu().numpy()
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
axes[1].set_title("Transformed Data (Px)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Transformed Feature 1", fontsize=12)
axes[1].set_ylabel("Transformed Feature 2", fontsize=12)
axes[1].legend()
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axes[1].set_aspect('equal', adjustable='box')
axes[1].tick_params(axis='both', which='major', labelsize=10)
axes[1].set_xlim(x_lim)
axes[1].set_ylim(y_lim)

plt.suptitle("Effect of Transformation P satisfying Cov(Px, z) = 0", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

# Save the plot
plot_save_path = os.path.join(output_dir, 'transformation_visualisation.png')
plt.savefig(plot_save_path)
print(f"Saved visualisation to {plot_save_path}")

plt.show()
