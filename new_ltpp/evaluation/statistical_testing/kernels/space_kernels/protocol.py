import torch
from typing import Protocol, runtime_checkable


@runtime_checkable
class ISpaceKernel(Protocol):
    @torch.compile
    def Gram_matrix(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """Compute the kernel matrix between two batches of sequences. This is used for the MMD mean term, where we need k(X^i_s,Y^j_t).

        Args:
            X: (B1, L, D)
            Y: (B2, K, D)
        Returns:
            matrix k(X^i_s,Y^j_t) of shape (B1, B2, length_X, length_Y)
        """
        ...

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix within a batch. Returns (B, L, L). This is used for the MMD variance term, where we need k(X^i_s,X^i_t) and k(Y^j_s,Y^j_t).

        Args:
            X: (B, L, D)
            Y: (B, K, D)
        Returns:
            matrix k(X^i_s,Y^i_t) of shape (B, length_X, length_Y)
        """
        ...
