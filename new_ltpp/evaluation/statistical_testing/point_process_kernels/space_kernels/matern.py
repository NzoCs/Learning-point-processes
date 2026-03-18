import torch
from .protocol import ISpaceKernel


class MaternTimeKernel(ISpaceKernel):
    def __init__(self, nu: float = 1.5, scaling: float = 1.0):
        self.nu = nu
        self.scaling = scaling
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("Only nu in {0.5,1.5,2.5} supported")

    def _compute_kernel(self, dist: torch.Tensor) -> torch.Tensor:
        if self.nu == 0.5:
            return torch.exp(-dist)
        elif self.nu == 1.5:
            scaled = torch.sqrt(torch.tensor(3.0)) * dist
            return (1 + scaled) * torch.exp(-scaled)
        else:
            scaled = torch.sqrt(torch.tensor(5.0)) * dist
            return (1 + scaled + scaled**2 / 3.0) * torch.exp(-scaled)

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
        return self.scaling * self._compute_kernel(dist / sigma)

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert X.shape[0] == Y.shape[0], "For batch kernel, X and Y must have the same batch size"
        B, L = X.shape
        X = X.unsqueeze(-1).expand(-1, L, -1)
        Y = Y.unsqueeze(-2).expand(-1, 1, L)
        dist = torch.sqrt((X - Y) ** 2)
        sigma = dist.median() + 1e-8  # Median heuristic for bandwidth
        return self.scaling * self._compute_kernel(dist / sigma)
