import torch
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from new_ltpp.shared_types import Batch, SimulationResult


class PointProcessKernel(ABC):
    """Abstract base class for point process kernels."""

    @abstractmethod
    def compute_gram_matrix(
        self,
        X: Batch | SimulationResult,
        Y: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Kernel function, takes two batches of sequences and returns a gram matrix.
        A Batch is either a Batch of training sequences or a Batch of simulated sequences.
        Batch and SimulationResult are just aliases for the same type.
        Args:
            phi_time_seqs: torch.Tensor (B1, N, L) N is the number of events in phi
            phi_type_seqs: torch.Tensor (B1, N, L)
            psi_time_seqs: torch.Tensor (B2, N, K) K is the number of events in psi
            psi_type_seqs: torch.Tensor (B2, N, K)
        Returns:
            torch.Tensor (B1, B2)
        """
        ...

    def compute_mmd(
        self, X: Batch | SimulationResult, Y: Batch | SimulationResult
    ) -> torch.Tensor:
        """Compute the MMD distance between two batches of sequences."""
        K_XY = self.compute_gram_matrix(X, Y)
        K_XX = self.compute_gram_matrix(X, X)
        K_YY = self.compute_gram_matrix(Y, Y)

        B, _ = K_XY.shape

        # Denominator for the unbiased estimator: max(B*(B-1), 1)
        # Computed in Python to avoid creating unnecessary tensor constants in the graph
        denom = float(max(B * (B - 1), 1))

        # Compute the MMD distance using the kernel values
        mmd_distance = K_XX.sum() / denom + K_YY.sum() / denom - 2 * K_XY.mean()
        return mmd_distance


@runtime_checkable
class IPointProcessKernel(Protocol):
    """Protocol for point process kernels (structural typing)."""

    def compute_gram_matrix(
        self,
        X: Batch | SimulationResult,
        Y: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Kernel function, takes two batches of sequences and returns a gram matrix."""
        ...

    def compute_mmd(
        self, X: Batch | SimulationResult, Y: Batch | SimulationResult
    ) -> torch.Tensor:
        """Compute the MMD distance between two batches of sequences.
        Args:
            X: First batch of sequences (B, L)
            Y: Second batch of sequences (B, K)
        Returns:
            The MMD distance as a torch.Tensor.
        """
        ...
