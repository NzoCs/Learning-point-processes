"""Base class for statistical tests with ABC + Protocol pattern.

Provides:
- ABC for inheritance and runtime @abstractmethod enforcement
- Protocol for IDE type checking and isinstance() checks
"""

from new_ltpp.configs import StatisticalTestConfig

from typing import Protocol, TypedDict, runtime_checkable
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels.kernel_protocol import (
    IPointProcessKernel,
)
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.models.model_protocol import ISimulableModel
from new_ltpp.shared_types import Batch


class TestStatistics(TypedDict):
    """Structured result for test statistics."""

    p_value: torch.Tensor
    observed_statistic: torch.Tensor
    permuted_statistics: torch.Tensor


class FinalTestResult(TypedDict):
    """Structured result for final test output."""

    p_value: float
    all_p_values: list[float]
    all_statistics: list[float]
    all_permuted_statistics: list[float]


@runtime_checkable
class ITest(Protocol):
    """Protocol for IDE type checking + isinstance() support."""

    kernel: IPointProcessKernel

    @property
    def name(self) -> str: ...

    def test_simulation(
        self,
        simulation: TypedDataLoader,
        ground_truth: TypedDataLoader,
    ) -> FinalTestResult:
        """Compute the p-value of the MMD two-sample permutation test for two data loaders, e.g. ground truth and simulation.

        Args:
            simulation: Data loader for simulated batches.
            ground_truth: Data loader for ground truth batches.
            accumulate: Whether to accumulate statistics.

        Returns:
            FinalTestResult containing final p-value, all p-values, and other statistics.
        """
        ...

    def test_model(
        self,
        model: ISimulableModel,
        data_loader: TypedDataLoader,
        statistical_test_config: StatisticalTestConfig,
    ) -> FinalTestResult:
        """Compute the p-value of the MMD two-sample permutation test for a model and a data loader, e.g. ground truth.
        It is recommended to use test_simulation when possible, as it allows to not simulate every time, rather use the pre-simulated batches.

        Args:
            model: ISimulableModel to simulate batches from.
            data_loader: Data loader for ground truth batches.
            accumulate: Whether to accumulate statistics.
        Returns:
            FinalTestResult containing final p-value, all p-values, and other statistics.
        """
        ...

    def compute_statistics(
        self,
        batch_x: Batch,
        batch_y: Batch,
        accumulate: bool = True,
    ) -> TestStatistics:
        """Compute the observed statistic, permuted statistics, and p-value for two batches of sequences.

        Args:
            batch_x: First batch of sequences (e.g. ground truth).
            batch_y: Second batch of sequences (e.g. simulation).
            accumulate: Whether to accumulate statistics.
        Returns:
            A TestStatistics typed dictionary containing the observed statistic, permuted statistics, and p-value.
        """
        ...
