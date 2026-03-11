import torch
from .protocol import ISpaceKernel


class LaplacianTimeKernel(ISpaceKernel):
    def __init__(self, scaling: float = 1.0):
        self.scaling = scaling

    @torch.compile
    def Gram_matrix(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        B1, L = X.shape
        B2, K = Y.shape
        X = X.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = Y.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        dist = torch.sqrt((X - Y) ** 2)
        sigma = dist.median() + 1e-8  # Median heuristic for bandwidth
        return self.scaling * torch.exp(-dist / sigma)

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        B, L = X.shape
        X = X.unsqueeze(-1).expand(-1, L, -1)
        Y = X.unsqueeze(-2).expand(-1, 1, L)
        dist = torch.sqrt((X - Y) ** 2)
        sigma = dist.median() + 1e-8  # Median heuristic for bandwidth
        return self.scaling * torch.exp(-dist / sigma)
