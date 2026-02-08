import torch
from typing import Protocol, runtime_checkable


@runtime_checkable
class ISpaceKernel(Protocol):
    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        """Compute the kernel matrix between two batches of sequences.
        Args:
            phi: (B1, L)
            psi: (B2, K)
        Returns:
            Tensor shaped (B1, B2, L, K)
        """
        ...

    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix within a batch. Returns (B, L, L)"""
        ...
