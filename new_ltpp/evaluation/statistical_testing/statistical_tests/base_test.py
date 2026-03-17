"""Base class for statistical tests with ABC + Protocol pattern.

Provides:
- ABC for inheritance and runtime @abstractmethod enforcement
- Protocol for IDE type checking and isinstance() checks
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import torch

from new_ltpp.evaluation.statistical_testing.kernels.kernel_protocol import (
    IPointProcessKernel,
)
from new_ltpp.models.base_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.shared_types import Batch


@runtime_checkable
class ITest(Protocol):
    """Protocol for IDE type checking + isinstance() support."""

    kernel: IPointProcessKernel

    @property
    def name(self) -> str: ...

    def p_value_from_model(
        self,
        model: NeuralModel,
        data_loader: TypedDataLoader,
    ) -> float: ...

    def p_value_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> torch.Tensor: ...

    def statistic_from_dataloaders(
        self,
        data_loader_x: TypedDataLoader,
        data_loader_y: TypedDataLoader,
    ) -> float: ...

    def statistic_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> torch.Tensor: ...

    def statistics_from_model(
        self,
        model: NeuralModel,
        data_loader: TypedDataLoader,
        accumulate: bool = True,
    ) -> tuple[float, list[float], list[float], list[float]]: ...

    def statistics_from_dataloaders(
        self,
        data_loader_x: TypedDataLoader,
        data_loader_y: TypedDataLoader,
        accumulate: bool = True,
    ) -> tuple[float, list[float], list[float], list[float]]: ...

    def statistics_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
        accumulate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class Test(ABC):
    """Abstract base class for statistical tests.

    Provides runtime enforcement via @abstractmethod.
    Concrete implementations must inherit from this class.
    """

    kernel: IPointProcessKernel

    def __init__(self, kernel: IPointProcessKernel):
        self.kernel = kernel

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the test."""
        pass

    @abstractmethod
    def statistics_from_model(
        self,
        model: NeuralModel,
        data_loader: TypedDataLoader,
    ) -> tuple[float, list[float], list[float], list[float]]:
        """Compute test statistics comparing model's distribution to data distribution."""
        pass

    @abstractmethod
    def statistics_from_dataloaders(
        self,
        data_loader_x: TypedDataLoader,
        data_loader_y: TypedDataLoader,
    ) -> tuple[float, list[float], list[float], list[float]]:
        """Compute test statistics comparing two data loaders."""
        pass

    @abstractmethod
    def statistics_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the test statistics comparing two batches."""
        pass

    # Convenience wrappers: extract a single value/statistic from the
    # more general `statistics_from_*` methods. These are used by scripts
    # and higher-level helpers that expect singular returns.
    def p_value_from_batches(self, batch_x: Batch, batch_y: Batch) -> torch.Tensor:
        p_value, _obs, _perms = self.statistics_from_batches(batch_x, batch_y)
        return p_value

    def statistic_from_batches(self, batch_x: Batch, batch_y: Batch) -> torch.Tensor:
        _p, observed, _perms = self.statistics_from_batches(batch_x, batch_y)
        return observed

    def p_value_from_model(
        self, model: NeuralModel, data_loader: TypedDataLoader
    ) -> float:
        p_val, _all_p, _all_mmds, _all_perms = self.statistics_from_model(
            model, data_loader
        )
        return p_val

    def statistic_from_dataloaders(
        self, data_loader_x: TypedDataLoader, data_loader_y: TypedDataLoader
    ) -> float:
        p_val, _all_p, _all_mmds, _all_perms = self.statistics_from_dataloaders(
            data_loader_x, data_loader_y
        )
        return p_val
