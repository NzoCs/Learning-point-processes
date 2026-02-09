import torch
from .protocol import ISpaceKernel


class LaplacianTimeKernel(ISpaceKernel):
    def __init__(self):
        pass

    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        B1, L = phi.shape
        B2, K = psi.shape
        X = phi.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = psi.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        dist = torch.sqrt(((X - Y) ** 2) + 1e-8)
        sigma = dist.median().item() + 1e-8  # Median heuristic for bandwidth
        return torch.exp(-dist / sigma)

    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        B, L = phi.shape
        X = phi.unsqueeze(-1).expand(-1, L, -1)
        Y = phi.unsqueeze(-2).expand(-1, 1, L)
        dist = torch.sqrt(((X - Y) ** 2) + 1e-8)
        sigma = dist.median().item() + 1e-8  # Median heuristic for bandwidth
        return torch.exp(-dist / sigma)
