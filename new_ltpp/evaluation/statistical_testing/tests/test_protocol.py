from typing import Protocol, runtime_checkable

from new_ltpp.evaluation.statistical_testing.kernels.kernel_protocol import (
    PointProcessKernel,
)
from new_ltpp.models.base_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.shared_types import Batch


@runtime_checkable
class TestProtocol(Protocol):
    kernel: PointProcessKernel

    def __init__(self, kernel: PointProcessKernel):
        self.kernel = kernel

    @property
    def name(self) -> str:
        """Return the name of the test."""
        ...

    def p_value_from_model(
        self,
        model: NeuralModel,
        data_loader: TypedDataLoader,
    ) -> float:
        """Compute the p-value of the test comparing the model's distribution to the data distribution.

        Args:
            model: The generative model to test.
            data_loader: DataLoader providing samples from the data distribution.
        Returns:
            p-value of the test.
        """
        ...

    def p_value_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> float:
        """Compute the p-value of the test comparing two batches of samples.

        Args:
            batch_x: Batch of samples from distribution X.
            batch_y: Batch of samples from distribution Y.
        Returns:
            p-value of the test.
        """
        ...

    def p_value_from_dataloaders(
        self,
        data_loader_x: TypedDataLoader,
        data_loader_y: TypedDataLoader,
    ) -> float:
        """Compute the p-value of the test comparing two data loaders.

        Args:
            data_loader_x: DataLoader providing samples from distribution X.
            data_loader_y: DataLoader providing samples from distribution Y.
        Returns:
            p-value of the test.
        """
        ...

    def get_statistic_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> float:
        """Compute the test statistic comparing two batches of samples.

        Args:
            batch_x: Batch of samples from distribution X.
            batch_y: Batch of samples from distribution Y.
        Returns:
            Test statistic value.
        """
        ...
