"""Base class for statistical metrics with ABC + Protocol pattern.

Provides:
- ABC for inheritance and runtime @abstractmethod enforcement
- Protocol (IStatMetric) for IDE type checking and isinstance() checks
"""

from abc import ABC, abstractmethod
from typing import Protocol

import torch

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.evaluation.statistical_testing.point_process_kernels.kernel_protocol import (
    IPointProcessKernel,
)


class IStatMetric(Protocol):
    """Protocol for IDE type checking + isinstance() support."""

    kernel: IPointProcessKernel

    def __call__(
        self, X: Batch | SimulationResult, Y: Batch | SimulationResult
    ) -> torch.Tensor: ...


class StatMetric(ABC):
    """Abstract base class for statistical metrics.

    Provides runtime enforcement via @abstractmethod.
    Concrete implementations must inherit from this class.
    """

    kernel: IPointProcessKernel

    def __init__(self, kernel: IPointProcessKernel) -> None:
        self.kernel = kernel

    def compute_mmd(
        self,
        X: Batch | SimulationResult,
        Y: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Compute the kernel matrix between two batches of sequences.

        Args:
            X: First batch of sequences (B, L)
            Y: Second batch of sequences (B, K)

        Returns:
            Kernel matrix of shape (B, B)
        """
        return self.kernel.compute_mmd(X, Y)

    @abstractmethod
    def __call__(
        self,
        X: Batch | SimulationResult,
        Y: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Compute the statistical metric between two batches.

        Args:
            X: First batch of sequences
            Y: Second batch of sequences

        Returns:
            The metric value as a torch.Tensor aggregated over the batch.
        """
        pass
