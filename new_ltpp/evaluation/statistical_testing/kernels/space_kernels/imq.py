import torch
from .protocol import ISpaceKernel


class IMQTimeKernel(ISpaceKernel):
    def __init__(self, c: float = 1.0, beta: float = 0.5):
        self.c = c
        self.beta = beta

    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        B1, L = phi.shape
        B2, K = psi.shape
        X = phi.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = psi.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        dist_sq = ((X - Y) ** 2).sum(dim=-1, keepdim=True)
        K = torch.pow(self.c**2 + dist_sq, -self.beta)
        return K

    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        B, L = phi.shape
        X = phi.unsqueeze(-1).expand(-1, L, -1)
        Y = phi.unsqueeze(-2).expand(-1, 1, L)
        dist_sq = ((X - Y) ** 2).sum(dim=-1)
        K = torch.pow(self.c**2 + dist_sq, -self.beta)
        return K
