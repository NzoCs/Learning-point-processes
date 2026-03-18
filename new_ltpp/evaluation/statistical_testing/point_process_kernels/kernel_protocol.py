import torch
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from new_ltpp.shared_types import Batch, SimulationResult


class PointProcessKernel(ABC):
    """Abstract base class for point process kernels."""
    
    @abstractmethod
    def compute_gram_matrix(
        self,
        phi_batch: Batch | SimulationResult,
        psi_batch: Batch | SimulationResult,
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
    
    def compute_mmd(self, phi: Batch | SimulationResult, psi: Batch | SimulationResult) -> torch.Tensor:
        """Compute the MMD distance between two batches of sequences."""
        K_XY = self.compute_gram_matrix(phi, psi)
        K_XX = self.compute_gram_matrix(phi, phi)
        K_YY = self.compute_gram_matrix(psi, psi)

        B, _ = K_XY.shape

        XX_reg = torch.max(
            torch.tensor(B * (B - 1), device=phi.time_seqs.device),
            torch.tensor(1.0, device=phi.time_seqs.device),
        )
        YY_reg = torch.max(
            torch.tensor(B * (B - 1), device=psi.time_seqs.device),
            torch.tensor(1.0, device=psi.time_seqs.device),
        )

        # Compute the MMD distance using the kernel values
        mmd_distance = K_XX.sum()/XX_reg + K_YY.sum()/YY_reg - 2 * K_XY.mean()
        return mmd_distance


@runtime_checkable
class IPointProcessKernel(Protocol):
    """Protocol for point process kernels (structural typing)."""
    
    def compute_gram_matrix(
        self,
        phi_batch: Batch | SimulationResult,
        psi_batch: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Kernel function, takes two batches of sequences and returns a gram matrix."""
        ...
    
    def compute_mmd(self, phi: Batch | SimulationResult, psi: Batch | SimulationResult) -> torch.Tensor:
        """Compute the MMD distance between two batches of sequences.
        Args: 
            phi: First batch of sequences (B, L)
            psi: Second batch of sequences (B, K)
        Returns:
            The MMD distance as a torch.Tensor.
        """
        ...

