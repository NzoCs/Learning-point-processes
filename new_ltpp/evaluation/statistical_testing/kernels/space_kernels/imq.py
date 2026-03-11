import torch
from .protocol import ISpaceKernel


class IMQTimeKernel(ISpaceKernel):
    def __init__(self, c: float = 1.0, beta: float = 0.5, scaling: float = 1.0):
        self.c = c
        self.beta = beta
        self.scaling = scaling

    @torch.compile
    def Gram_matrix(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        B1, L = X.shape
        B2, K = Y.shape
        X = X.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = Y.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        dist_sq = (X - Y) ** 2
        sigma = dist_sq.median() + 1e-8  # Median heuristic for bandwidth
        K = torch.pow(self.c**2 + dist_sq / sigma, -self.beta)
        return self.scaling * K

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        B, L = X.shape
        X = X.unsqueeze(-1).expand(-1, L, -1)
        Y = Y.unsqueeze(-2).expand(-1, 1, L)
        dist_sq = (X - Y) ** 2
        sigma = dist_sq.median() + 1e-8  # Median heuristic for bandwidth
        K = torch.pow(self.c**2 + dist_sq / sigma, -self.beta)
        return self.scaling * K
