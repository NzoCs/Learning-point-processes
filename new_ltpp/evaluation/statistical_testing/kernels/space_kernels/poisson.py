import torch
from .protocol import ISpaceKernel


class PoissonTimeKernel(ISpaceKernel):
    def __init__(self, r: float = 0.5, d: int = 1):
        if not (0 < r < 1):
            raise ValueError("Parameter r must be in (0,1)")
        self.r = r
        self.d = d

    @torch.compile
    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        B1, L = phi.shape
        B2, K = psi.shape
        X = phi.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = psi.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        inner = X * Y
        num = 1 - self.r**2
        denom = torch.pow(1 - 2 * self.r * inner + self.r**2, self.d / 2.0)
        return num / (denom + 1e-8)

    @torch.compile
    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        B, L = phi.shape
        X = phi.unsqueeze(-1).expand(-1, L, -1)
        Y = phi.unsqueeze(-2).expand(-1, 1, L)
        inner = X * Y
        num = 1 - self.r**2
        denom = torch.pow(1 - 2 * self.r * inner + self.r**2, self.d / 2.0)
        return num / (denom + 1e-8)
