import torch
from .protocol import ISpaceKernel


class RBFTimeKernel(ISpaceKernel):
    def __init__(self):
        pass

    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        B1, L = phi.shape
        B2, K = psi.shape

        X = phi.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = psi.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)

        norm_matrix = ((X - Y) ** 2)
        sigma = norm_matrix.median().item() ** 0.5 + 1e-8  # Median heuristic for bandwidth 
        Kmat = torch.exp(-norm_matrix / (2 * sigma**2))
        return Kmat

    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        B, L = phi.shape
        X = phi.unsqueeze(-1).expand(-1, L, -1)
        Y = phi.unsqueeze(-2).expand(-1, 1, L)
        norm_matrix = ((X - Y) ** 2)
        sigma = norm_matrix.median().item() + 1e-8  # Median heuristic for bandwidth 
        K = torch.exp(-norm_matrix / (2 * sigma))
        return K
