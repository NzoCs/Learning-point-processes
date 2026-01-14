import torch
from sigkernel import SigKernel, LinearKernel, RBFKernel
from enum import Enum
from typing import Optional, TypedDict

from new_ltpp.shared_types import Batch, SimulationResult
from .kernel_abc import KernelABC


class StaticKernelEnum(Enum):
    LinearKernel = "linear"
    RBFKernel = "rbf"


class SequenceEmbeddingEnum(Enum):
    LinearInterpolant = "linear_interpolant"
    ConstantInterpolant = "constant_interpolant"


class Embedding(TypedDict):
    time_seqs: torch.Tensor
    type_seqs: torch.Tensor
    counting_seqs: torch.Tensor


class SIGKernel(KernelABC):
    def __init__(
        self,
        static_kernel_type: StaticKernelEnum,
        embedding_type: SequenceEmbeddingEnum,
        dyadic_order: int,
        sigma: Optional[float] = None,
    ):
        match static_kernel_type:
            case StaticKernelEnum.LinearKernel:
                self.kernel = SigKernel(
                    static_kernel=LinearKernel(), dyadic_order=dyadic_order
                )
            case StaticKernelEnum.RBFKernel:
                self.kernel = SigKernel(
                    static_kernel=RBFKernel(sigma=sigma), dyadic_order=dyadic_order
                )

        self.embedding_type = embedding_type

    def _get_embedding(self, phi: Batch | SimulationResult) -> Embedding:
        """Get the embedding of the sequences.
        Args:
            phi: Batch of sequences of shape (batch_size, seq_len)
        Returns:
            Embedding for the computation of the kernel via the SIG kernel
        """

        # Ne gère pas encore les masques à implementer !

        match self.embedding_type:
            case SequenceEmbeddingEnum.LinearInterpolant:
                time_seqs = phi.time_seqs
                type_seqs = phi.type_seqs

                # The first event is an event so counting seq is [1, ...., N+1]
                counting_seqs = (
                    torch.arange(
                        phi.time_seqs.shape[1], device=phi.time_seqs.device
                    ).expand(phi.time_seqs.shape[0], -1)
                    + 1
                )

                return Embedding(
                    time_seqs=time_seqs,
                    type_seqs=type_seqs,
                    counting_seqs=counting_seqs,
                )
            case SequenceEmbeddingEnum.ConstantInterpolant:
                # create the sequence of [t_1, t_1, t_2, t_2, ..., t_n, t_n]
                pair_indexes = (
                    torch.arange(
                        phi.time_seqs.shape[1], device=phi.time_seqs.device
                    ).expand(phi.time_seqs.shape[0], -1)
                    * 2
                )
                time_seqs = torch.zeros(
                    (phi.time_seqs.shape[0], phi.time_seqs.shape[1] * 2)
                )
                time_seqs[:, pair_indexes] = phi.time_seqs
                time_seqs[:, pair_indexes + 1] = phi.time_seqs

                # same thing for the type and counting sequences
                type_seqs = torch.zeros(
                    (phi.time_seqs.shape[0], phi.time_seqs.shape[1] * 2)
                )

                type_seqs[:, pair_indexes] = phi.type_seqs
                type_seqs[:, pair_indexes + 1] = phi.type_seqs

                counting_seqs = (
                    torch.arange(
                        phi.time_seqs.shape[1], device=phi.time_seqs.device
                    ).expand(phi.time_seqs.shape[0], -1)
                    + 1
                )

                padded_counting_seqs = torch.zeros(
                    (phi.time_seqs.shape[0], phi.time_seqs.shape[1] * 2)
                )
                padded_counting_seqs[:, pair_indexes] = counting_seqs
                padded_counting_seqs[:, pair_indexes + 1] = counting_seqs

                return Embedding(
                    time_seqs=time_seqs,
                    type_seqs=type_seqs,
                    counting_seqs=padded_counting_seqs,
                )
            case _:
                raise ValueError(f"Unknown embedding type: {self.embedding_type}")

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

        phi_embedding, psi_embedding = (
            self._get_embedding(phi),
            self._get_embedding(psi),
        )

        return self.kernel.compute_kernel(phi_embedding, psi_embedding)
