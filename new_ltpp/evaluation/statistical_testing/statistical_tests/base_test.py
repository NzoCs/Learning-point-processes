"""Base class for statistical tests with ABC + Protocol pattern.

Provides:
- ABC for inheritance and runtime @abstractmethod enforcement
- Protocol for IDE type checking and isinstance() checks
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

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
    ) -> float: ...

    def p_value_from_dataloaders(
        self,
        data_loader_x: TypedDataLoader,
        data_loader_y: TypedDataLoader,
    ) -> float: ...

    def statistic_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> float: ...


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
    def p_value_from_model(
        self,
        model: NeuralModel,
        data_loader: TypedDataLoader,
    ) -> float:
        """Compute p-value comparing model's distribution to data distribution."""
        pass

    @abstractmethod
    def p_value_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> float:
        """Compute p-value comparing two batches of samples."""
        pass

    @abstractmethod
    def p_value_from_dataloaders(
        self,
        data_loader_x: TypedDataLoader,
        data_loader_y: TypedDataLoader,
    ) -> float:
        """Compute p-value comparing two data loaders."""
        pass

    @abstractmethod
    def statistic_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> float:
        """Compute the test statistic comparing two batches."""
        pass
