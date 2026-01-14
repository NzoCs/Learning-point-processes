import torch
from abc import ABC, abstractmethod

from new_ltpp.shared_types import Batch, SimulationResult


class KernelABC(ABC):
    @abstractmethod
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
        pass
