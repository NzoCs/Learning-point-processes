import torch
from .protocol import ISpaceKernel


class RationalQuadraticTimeKernel(ISpaceKernel):
    def __init__(self, sigma: float = 1.0, alpha: float = 1.0):
        self.sigma = sigma
        self.alpha = alpha

    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        B1, L = phi.shape
        B2, K = psi.shape
        X = phi.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = psi.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        dist_sq = ((X - Y) ** 2).sum(dim=-1, keepdim=True)
        K = torch.pow(1 + dist_sq / (2 * self.alpha * self.sigma**2), -self.alpha)
        return K

    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        B, L = phi.shape
        X = phi.unsqueeze(-1).expand(-1, L, -1)
        Y = phi.unsqueeze(-2).expand(-1, 1, L)
        dist_sq = ((X - Y) ** 2).sum(dim=-1, keepdim=True)
        return torch.pow(1 + dist_sq / (2 * self.alpha * self.sigma**2), -self.alpha)
