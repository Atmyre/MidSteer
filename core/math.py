import typing as tp
import torch


def fractional_matrix_power_cov_torch(mat: torch.Tensor, alpha: float, eps=1e-10) -> torch.Tensor:
    device = mat.device
    if mat.device.type == 'mps':  # Workaround because MPS does not yet support torch.linalg.eig
        mat = mat.cpu()

    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clip(evals, min=0, max=None)
    evals = torch.where(evals >= eps, evals ** alpha, 0.)
    return (evecs @ torch.diag_embed(evals) @ evecs.mT).to(device)


def convert_to_widest_dtype(vector: torch.Tensor, device: tp.Any, force_double: bool = False):
    # float64 is needed for numerical stability
    if device.type == 'mps':
        if force_double:
            return vector.to('cpu').to(dtype=torch.float64)
        else:
            return vector.to(device, dtype=torch.float32)
    else:
        return vector.to(device, dtype=torch.float64)