"""Base class for statistical metrics with ABC + Protocol pattern.

Provides:
- ABC for inheritance and runtime @abstractmethod enforcement
- Protocol (IStatMetric) for IDE type checking and isinstance() checks
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.evaluation.statistical_testing.kernels.kernel_protocol import (
    IPointProcessKernel,
)


@runtime_checkable
class IStatMetric(Protocol):
    """Protocol for IDE type checking + isinstance() support."""

    kernel: IPointProcessKernel

    def compute_kernel_matrix(
        self,
        phi: Batch | SimulationResult,
        psi: Batch | SimulationResult,
    ) -> torch.Tensor: ...

    def __call__(
        self,
        phi: Batch | SimulationResult,
        psi: Batch | SimulationResult,
    ) -> float: ...


class StatMetric(ABC):
    """Abstract base class for statistical metrics.

    Provides runtime enforcement via @abstractmethod.
    Concrete implementations must inherit from this class.
    """

    kernel: IPointProcessKernel

    def __init__(self, kernel: IPointProcessKernel) -> None:
        self.kernel = kernel

    def compute_kernel_matrix(
        self,
        phi: Batch | SimulationResult,
        psi: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Compute the kernel matrix between two batches of sequences.

        Args:
            phi: First batch of sequences
            psi: Second batch of sequences

        Returns:
            Kernel matrix of shape (B1, B2)
        """
        return self.kernel.compute_gram_matrix(phi, psi)

    @abstractmethod
    def __call__(
        self,
        phi: Batch | SimulationResult,
        psi: Batch | SimulationResult,
    ) -> float:
        """Compute the statistical metric between two batches.

        Args:
            phi: First batch of sequences
            psi: Second batch of sequences

        Returns:
            The metric value as a float aggregated over the batch.
        """
        pass
