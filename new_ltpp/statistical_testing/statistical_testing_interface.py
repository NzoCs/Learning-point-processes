from polars.dependencies import torch
from typing import Protocol
from new_ltpp.models.neural_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.shared_types import Batch, SimulationResult


class StatisticalTest(Protocol):
    """Statistical test for goodness of fit of point process models."""

    def __init__(self): ...

    def test(self, model: NeuralModel, dataset: TypedDataLoader) -> bool:
        """Perform a goodness of fit test. The null hypothesis is that the model is a good fit for the data.
        If we fail to reject the null hypothesis, we can say that the model is a good fit for the data.
        Args:
            model: The model to test.
            dataset: The dataset to test on.
        Returns:
            False if the null hypothesis is rejected, True otherwise.
        """
        ...

    def p_value(self, model: NeuralModel, dataset: TypedDataLoader) -> float:
        """Compute the p-value of the test.
        Args:
            model: The model to test.
            dataset: The dataset to test on.
        Returns:
            The p-value of the test.
        """
        ...


class Kernel(Protocol):
    """Kernel Protocol, the kernel is called as a function on a pair of batches,
    returns a batch of kernel values."""

    def __call__(
        self, phi: Batch | SimulationResult, psi: Batch | SimulationResult
    ) -> torch.Tensor:
        """Kernel function, takes two batches of sequences and returns a batch of kernel values.
        A Batch is either a Batch of training sequences or a Batch of simulated sequences.
        Batch and SimulationResult are just aliases for the same type.
        Args:
            phi: Batch of sequences of shape (batch_size, seq_len)
            psi: Batch of sequences of shape (batch_size, seq_len)
        Returns:
            Batch of kernel values of shape (batch_size)
        """
        ...
