from typing import Protocol, TypedDict
import torch
from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.evaluation.statistical_testing.kernels.kernel_protocol import (
    PointProcessKernel,
)


class NewPoints(TypedDict):
    time_deltas: torch.Tensor  # shape: (batch_size, N_samples)
    times: torch.Tensor  # shape: (batch_size, N_samples)
    types: torch.Tensor  # shape: (batch_size, N_samples)


class StatMetricsProtocol(Protocol):
    kernel: PointProcessKernel

    def __init__(
        self,
        kernel: PointProcessKernel,
    ) -> None:
        self.kernel = kernel

    def compute_kernel_matrix(
        self,
        phi: Batch | SimulationResult,
        psi: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Compute the kernel matrix between two batches of sequences.

        Args:
            phi_time_seqs: (B1, L) N is the number of events in phi
            phi_type_seqs: (B1, L)
            psi_time_seqs: (B2, K) K is the number of events in psi
            psi_type_seqs: (B2, K)

        Returns:
            Kernel matrix of shape (B1, B2)
        """

        return self.kernel.graam_matrix(phi, psi)  # (B1, B2)

    def __call__(
        self, phi: Batch | SimulationResult, psi: Batch | SimulationResult
    ) -> float:
        """Compute the statistical metric between the model's simulated sequences and the real sequences in the batch.
        args:
            model: The neural point process model to evaluate.
            batch: Batch of real sequences.
        returns:
            The metric value as a float aggregated over the batch.
        """
        ...
