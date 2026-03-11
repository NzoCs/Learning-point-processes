from typing import Optional

import torch
from .protocol import ISpaceKernel


class RationalQuadraticTimeKernel(ISpaceKernel):
    def __init__(self, sigma: Optional[float] = None, alpha: float = 0.8, scaling: float = 1.0):
        self.alpha = alpha
        self.scaling = scaling
        self.sigma = sigma

    @torch.compile
    def Gram_matrix(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        B1, L = X.shape
        B2, K = Y.shape
        X = X.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = Y.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        dist_sq = (X - Y) ** 2

        sigma = self.sigma if self.sigma is not None else dist_sq.median()  # Median heuristic for bandwidth
        mat = torch.pow(1 + dist_sq / (2 * self.alpha * sigma), -self.alpha)
        return self.scaling * mat

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert X.shape[0] == Y.shape[0], "For batch kernel, X and Y must have the same batch size"
        B, L = X.shape
        X = X.unsqueeze(-1).expand(-1, L, -1)
        Y = Y.unsqueeze(-2).expand(-1, 1, L)
        dist_sq = (X - Y) ** 2

        sigma = self.sigma if self.sigma is not None else dist_sq.median()  # Median heuristic for bandwidth
        return self.scaling * torch.pow(1 + dist_sq / (2 * self.alpha * sigma), -self.alpha)
