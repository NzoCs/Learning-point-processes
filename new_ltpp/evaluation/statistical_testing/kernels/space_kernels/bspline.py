import torch
from .protocol import ISpaceKernel


class BSplineTimeKernel(ISpaceKernel):
    def __init__(self, order: int = 3):
        self.order = order

    def _bspline_kernel(self, t: torch.Tensor) -> torch.Tensor:
        if self.order == 2:
            result = torch.zeros_like(t)
            mask1 = (t >= 0) & (t < 1)
            mask2 = (t >= 1) & (t < 2)
            result[mask1] = 0.75 - t[mask1] ** 2
            result[mask2] = 0.5 * (2 - t[mask2]) ** 2
            return result
        elif self.order == 3:
            result = torch.zeros_like(t)
            mask1 = (t >= 0) & (t < 1)
            mask2 = (t >= 1) & (t < 2)
            result[mask1] = 2.0 / 3.0 - t[mask1] ** 2 + 0.5 * t[mask1] ** 3
            result[mask2] = (2 - t[mask2]) ** 3 / 6.0
            return result
        else:
            raise ValueError(f"Unsupported B-spline order: {self.order}")

    @torch.compile
    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        B1, L = phi.shape
        B2, K = psi.shape
        X = phi.unsqueeze(-1).unsqueeze(1).expand(-1, B2, -1, 1)
        Y = psi.unsqueeze(-2).unsqueeze(0).expand(B1, -1, 1, -1)
        dist = torch.abs(X - Y)
        sigma = dist.median() + 1e-8  # Median heuristic for bandwidth
        normalized = dist / sigma
        return self._bspline_kernel(normalized)

    @torch.compile
    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        B, L = phi.shape
        X = phi.unsqueeze(-1).expand(-1, L, -1)
        Y = phi.unsqueeze(-2).expand(-1, 1, L)
        dist = torch.abs(X - Y)
        sigma = dist.median() + 1e-8  # Median heuristic for bandwidth
        normalized = dist / sigma
        return self._bspline_kernel(normalized)
