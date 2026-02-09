import torch
from .protocol import ISpaceKernel


class RationalQuadraticTimeKernel(ISpaceKernel):
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha

    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        B1, L = phi.shape
        B2, K = psi.shape
        X = phi.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = psi.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        dist_sq = ((X - Y) ** 2)

        sigma = dist_sq.median().item() + 1e-8  # Median heuristic for bandwidth
        K = torch.pow(1 + dist_sq / (2 * self.alpha * sigma), -self.alpha)
        return K

    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        B, L = phi.shape
        X = phi.unsqueeze(-1).expand(-1, L, -1)
        Y = phi.unsqueeze(-2).expand(-1, 1, L)
        dist_sq = ((X - Y) ** 2)
        sigma = dist_sq.median().item() + 1e-8  # Median heuristic for bandwidth
        return torch.pow(1 + dist_sq / (2 * self.alpha * sigma), -self.alpha)
