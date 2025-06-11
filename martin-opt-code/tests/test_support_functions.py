import unittest
import torch
import os
import torch.nn.functional as F

# Assuming src is in the Python path or using relative imports
from src.support_functions import empirical_cross_covariance_with_torch_cov

# Direct implementation for comparison
def empirical_cross_covariance_direct(x, y, unbiased=True):
    """
    Computes the empirical cross-covariance matrix between two sets of samples.
    (Direct implementation for testing purposes)
    """
    N = x.shape[0]
    if y.shape[0] != N:
        raise ValueError("x and y must have the same number of samples (dim 0)")
    if N < 2 and unbiased:
        raise ValueError("Cannot compute unbiased estimate with N < 2 samples.")
    if N < 1:
        raise ValueError("Cannot compute covariance with N < 1 samples.")

    x_mean = torch.mean(x, dim=0, keepdim=True)
    y_mean = torch.mean(y, dim=0, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    cov_sum = x_centered.T @ y_centered

    if unbiased:
        denominator = N - 1
    else:
        denominator = N

    cov_xy = cov_sum / denominator
    return cov_xy

class TestSupportFunctions(unittest.TestCase):

    def test_cross_covariance(self):
        """
        Tests if empirical_cross_covariance_with_torch_cov matches
        the direct implementation.
        """
        N = 100
        d_x = 5
        d_y = 3
        # Use double precision for better numerical stability in tests
        x_samples = torch.randn(N, d_x, dtype=torch.float64)
        y_samples = torch.randn(N, d_y, dtype=torch.float64) * 0.5 + x_samples[:, :d_y] # Correlated data

        # Test unbiased estimate (correction=1)
        cov_torch_unbiased = empirical_cross_covariance_with_torch_cov(x_samples, y_samples, correction=1)
        cov_direct_unbiased = empirical_cross_covariance_direct(x_samples, y_samples, unbiased=True)
        self.assertTrue(torch.allclose(cov_torch_unbiased, cov_direct_unbiased, atol=1e-7),
                        "Unbiased estimates do not match")

        # Test biased estimate (correction=0)
        cov_torch_biased = empirical_cross_covariance_with_torch_cov(x_samples, y_samples, correction=0)
        cov_direct_biased = empirical_cross_covariance_direct(x_samples, y_samples, unbiased=False)
        self.assertTrue(torch.allclose(cov_torch_biased, cov_direct_biased, atol=1e-7),
                        "Biased estimates do not match")

    def test_edge_cases(self):
        """ Tests edge cases like small N """
        d_x = 2
        d_y = 1
        x_2 = torch.randn(2, d_x, dtype=torch.float64)
        y_2 = torch.randn(2, d_y, dtype=torch.float64)

        # Unbiased with N=2 should work
        try:
            empirical_cross_covariance_with_torch_cov(x_2, y_2, correction=1)
        except ValueError:
            self.fail("Unbiased estimate failed unexpectedly for N=2")

        # Biased with N=1 should work
        x_1 = torch.randn(1, d_x, dtype=torch.float64)
        y_1 = torch.randn(1, d_y, dtype=torch.float64)
        try:
            cov = empirical_cross_covariance_with_torch_cov(x_1, y_1, correction=0)
            # Biased estimate with N=1 should be all zeros
            self.assertTrue(torch.allclose(cov, torch.zeros_like(cov)))
        except ValueError:
            self.fail("Biased estimate failed unexpectedly for N=1")

        # Unbiased with N=1 should raise ValueError
        with self.assertRaises(ValueError):
            empirical_cross_covariance_with_torch_cov(x_1, y_1, correction=1)

        # Mismatched N should raise ValueError
        x_mismatch = torch.randn(5, d_x, dtype=torch.float64)
        y_mismatch = torch.randn(4, d_y, dtype=torch.float64)
        with self.assertRaises(ValueError):
            empirical_cross_covariance_with_torch_cov(x_mismatch, y_mismatch)

    def test_p_optimal_constraint(self):
        """
        Tests if the loaded P_optimal satisfies Cov(Px, z) = 0,
        where z is the one-hot encoding of the labels.
        """
        # Define paths relative to the project root
        project_root = os.path.dirname(os.path.dirname(__file__)) # Assumes tests/ is one level down from root
        p_path = os.path.join(project_root, 'results', 'P_optimal.pt')
        x_path = os.path.join(project_root, 'data', 'x_data.pt')
        z_path = os.path.join(project_root, 'data', 'z_labels.pt')

        # Check if files exist
        self.assertTrue(os.path.exists(p_path), f"File not found: {p_path}")
        self.assertTrue(os.path.exists(x_path), f"File not found: {x_path}")
        self.assertTrue(os.path.exists(z_path), f"File not found: {z_path}")

        # Load data
        P = torch.load(p_path)
        x = torch.load(x_path)
        z = torch.load(z_path)

        # Ensure data is float64 for precision, similar to other tests
        P = P.to(torch.float64)
        x = x.to(torch.float64)
        # Keep z as LongTensor for one-hot encoding

        # Ensure z is 1D LongTensor for one-hot encoding
        if z.ndim != 1:
            # Assuming z might be loaded as [N, 1], squeeze it to [N]
            if z.shape[1] == 1:
                z = z.squeeze(1)
            else:
                self.fail(f"Expected z_labels.pt to contain 1D tensor or Nx1 tensor, but got shape {z.shape}")
        z = z.long() # Ensure it's LongTensor

        # Perform one-hot encoding
        num_classes = int(z.max().item() + 1)
        z_one_hot = F.one_hot(z, num_classes=num_classes).to(torch.float64)
        # z_one_hot will have shape (N, num_classes)

        # Apply transformation P
        # Assuming x is (N, d_x) and P is (d_out, d_x)
        # We want Px, which means applying P to each x vector.
        # P @ x_i^T for each row x_i. This is equivalent to x @ P.T
        Px = x @ P.T # Shape (N, d_out)

        # Compute cross-covariance Cov(Px, z_one_hot)
        cov_Px_z = empirical_cross_covariance_with_torch_cov(Px, z_one_hot, correction=1) # Use unbiased

        # Check if the covariance is close to zero
        self.assertTrue(torch.allclose(cov_Px_z, torch.zeros_like(cov_Px_z), atol=1e-6), # Increased tolerance
                        f"Cov(Px, z_one_hot) is not close to zero. Value:\n{cov_Px_z}")


if __name__ == '__main__':
    unittest.main()
