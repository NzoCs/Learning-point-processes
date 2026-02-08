import torch
from typing import Protocol, runtime_checkable

from new_ltpp.shared_types import Batch, SimulationResult


@runtime_checkable
class IPointProcessKernel(Protocol):
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
