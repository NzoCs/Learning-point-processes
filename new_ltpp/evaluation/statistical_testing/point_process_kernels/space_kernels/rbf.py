import torch
from typing import Optional
from .protocol import ISpaceKernel


class RBFTimeKernel(ISpaceKernel):
    def __init__(
        self,
        sigma: Optional[float] = None,
        scaling: float = 1.0,
    ):
        self.scaling = scaling
        self.sigma = sigma

    @torch.compile
    def Gram_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        B1, L = X.shape
        B2, K = Y.shape

        X = X.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = Y.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)

        norm_matrix = (X - Y) ** 2
        sigma = (
            self.sigma if self.sigma is not None else norm_matrix.median() + 1e-8
        )  # Median heuristic for bandwidth
        Kmat = torch.exp(-norm_matrix / (2 * sigma))
        return self.scaling * Kmat

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert X.shape[0] == Y.shape[0], (
            "For batch kernel, X and Y must have the same batch size"
        )

        B, LX = X.shape
        _, LY = Y.shape
        X = X.unsqueeze(-1).expand(-1, -1, LY)
        Y = Y.unsqueeze(-2).expand(-1, LX, -1)
        norm_matrix = (X - Y) ** 2
        sigma = (
            self.sigma if self.sigma is not None else norm_matrix.median() + 1e-8
        )  # Median heuristic for bandwidth
        mat = torch.exp(-norm_matrix / (2 * sigma))
        return self.scaling * mat
