"""
Script to solve the optimization problem:

min_{A, b} E_x[ || Ax + b - x ||_M^2 ] subject to Cov(Ax + b, z) = -Cov(x, z)
which simplifies to (A + I) Sigma_xz = 0

using the Condat-Vu primal-dual algorithm.

Algorithm Steps:
1. Load data x (features) and z (labels) from the 'data/' directory.
2. Convert labels z into one-hot encoded vectors.
3. Compute covariance matrix Sigma_xx = Cov(x, x), cross-covariance
   matrix Sigma_xz = Cov(x, z_one_hot), and mean E_x[x] using functions from
   'src/support_functions.py' or standard torch functions.
4. Initialise the metric M (default: identity matrix).
5. Initialise the optimization variables A (e.g., identity matrix),
   b (e.g., zero vector), and Lambda (dual variable, e.g., zero matrix).
6. Calculate operator norms ||M||, ||Sigma_xx||, ||Sigma_xz||.
7. Determine appropriate step sizes t (primal) and s (dual) based on
   Condat-Vu stability conditions (using approximations based on A).
8. Iterate using the Condat-Vu update rules:
   A_k = A_{k-1} - t * ( M(A_{k-1} - I)Sigma_xx + M b_{k-1}E_x[x]^T + Lambda_{k-1} Sigma_xz^T )
   b_k = b_{k-1} - t * M(b_{k-1} + (A_{k-1} - I)E_x[x])
   Lambda_k = Lambda_{k-1} + s * ( (2*A_k - A_{k-1} + I) @ Sigma_xz )
9. Check the stopping criterion based on the average norm of the
   optimality conditions (residuals):
   - Primal residual A: M(A - I)Sigma_xx + M b E_x[x]^T + Lambda Sigma_xz^T
   - Primal residual b: M(b + (A - I)E_x[x])
   - Dual residual: (A + I) Sigma_xz
   Stop when avg_norm(residuals) < TOLERANCE.
10. Print the final matrices A, b, residual norms, and iteration count.
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

    # Calculate E[x]
    mean_x = torch.mean(x, dim=0, keepdim=True).T # Shape (n, 1)
    print(f"Computed Mean(x) (mean_x) with shape: {mean_x.shape}")

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
A = torch.eye(n, dtype=x.dtype, device=x.device) # Initial A = Identity
b = torch.zeros(n, 1, dtype=x.dtype, device=x.device) # Initial b = 0 vector
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
    print(f"\n--- Step Sizes ---")
    print(f"Fixed s = {s:.4e}")

    # Calculate t based on the condition: t < 1 / (||M||*||Sigma_xx|| + s*||Sigma_xz||^2)
    # Note: This is an approximation, ignoring terms related to b and mean_x for simplicity.
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
    A_prev = A.clone()
    b_prev = b.clone()
    Lambda_prev = Lambda.clone()
    I_n = torch.eye(n, dtype=A.dtype, device=A.device) # Identity matrix

    # Update A
    grad_A = M @ (A_prev - I_n) @ sigma_xx + M @ b_prev @ mean_x.T + Lambda_prev @ sigma_xz.T
    A = A_prev - t * grad_A

    # Update b
    grad_b = M @ (b_prev + (A_prev - I_n) @ mean_x)
    b = b_prev - t * grad_b

    # Update Lambda
    A_extrapolated = 2 * A - A_prev
    grad_Lambda = (A_extrapolated + I_n) @ sigma_xz # Constraint is (A+I)Sigma_xz = 0
    Lambda = Lambda_prev + s * grad_Lambda

    # Check stopping criterion (Optimality conditions)
    primal_residual_A = M @ (A - I_n) @ sigma_xx + M @ b @ mean_x.T + Lambda @ sigma_xz.T
    primal_residual_b = M @ (b + (A - I_n) @ mean_x)
    dual_residual = (A + I_n) @ sigma_xz

    norm_primal_res_A = torch.linalg.norm(primal_residual_A).item()
    norm_primal_res_b = torch.linalg.norm(primal_residual_b).item()
    norm_dual_res = torch.linalg.norm(dual_residual).item()
    # Combine residuals for stopping check
    avg_residual_norm = (norm_primal_res_A + norm_primal_res_b + norm_dual_res) / 3.0

    if (k + 1) % 10 == 0:
        print(f"Iter {k+1}/{MAX_ITER}, Avg Residual Norm: {avg_residual_norm:.4e} (A:{norm_primal_res_A:.2e}, b:{norm_primal_res_b:.2e}, L:{norm_dual_res:.2e})")

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
print(f"Final Matrix A (Shape: {A.shape}):")
# print(A) # Optionally print the full matrix
print(f"Final Vector b (Shape: {b.shape}):")
# print(b) # Optionally print the full vector
print(f"Final Lambda (Shape: {Lambda.shape}):")
# print(Lambda) # Optionally print the full matrix
print(f"Final Primal Residual A Norm: {norm_primal_res_A:.4e}")
print(f"Final Primal Residual b Norm: {norm_primal_res_b:.4e}")
print(f"Final Dual Residual Norm (Constraint Violation Norm): {norm_dual_res:.4e}")
print(f"Final Average Residual Norm: {avg_residual_norm:.4e}")

# --- Optional: Save the result ---
output_dir = os.path.join(project_root, 'results')
os.makedirs(output_dir, exist_ok=True)
A_save_path = os.path.join(output_dir, 'A_optimal.pt')
b_save_path = os.path.join(output_dir, 'b_optimal.pt')
torch.save(A, A_save_path)
torch.save(b, b_save_path)
print(f"\nSaved optimal matrix A to {A_save_path}")
print(f"Saved optimal vector b to {b_save_path}")

# --- Visualise the Transformation ---
print("\n--- Visualising Transformation ---")

# Load the original data (already loaded as x, z)
# Load the computed A and b matrices (already in memory)

# Apply the transformation Ax + b to x
# Ensure A and b are on the same device and dtype as x if necessary
A = A.to(x.device, dtype=x.dtype)
b = b.to(x.device, dtype=x.dtype)
# x is N x n, A is n x n, b is n x 1
# x @ A.T is N x n
# b.T is 1 x n
x_transformed = x @ A.T + b.T # Apply transformation: x_new = Ax + b

# Convert to numpy for plotting
x_np = x.cpu().numpy()
x_transformed_np = x_transformed.cpu().detach().numpy() # Detach if A or b requires grad
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
axes[1].scatter(x_transformed_np[z_np == 0, 0], x_transformed_np[z_np == 0, 1],
                c='orange', label='Label 0', alpha=0.5, edgecolors='k', linewidth=0.5)
axes[1].scatter(x_transformed_np[z_np == 1, 0], x_transformed_np[z_np == 1, 1],
                c='blue', label='Label 1', alpha=0.5, edgecolors='k', linewidth=0.5)
axes[1].set_title("Transformed Data (Ax + b)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Transformed Feature 1", fontsize=12)
axes[1].set_ylabel("Transformed Feature 2", fontsize=12)
axes[1].legend()
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axes[1].set_aspect('equal', adjustable='box')
axes[1].tick_params(axis='both', which='major', labelsize=10)
axes[1].set_xlim(x_lim)
axes[1].set_ylim(y_lim)

plt.suptitle("Effect of Transformation Ax + b satisfying Cov(Ax+b, z) = -Cov(x, z)", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

# Save the plot
plot_save_path = os.path.join(output_dir, 'new_transformation_visualisation_with_bias.png')
plt.savefig(plot_save_path)
print(f"Saved visualisation to {plot_save_path}")

plt.show()