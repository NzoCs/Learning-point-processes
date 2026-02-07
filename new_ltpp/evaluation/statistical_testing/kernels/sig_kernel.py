import torch
from sigkernel import SigKernel, LinearKernel, RBFKernel
from typing import Optional, TypedDict, Literal

from .kernel_protocol import PointProcessKernel
from new_ltpp.shared_types import Batch, SimulationResult


class Embedding(TypedDict):
    time_seqs: torch.Tensor
    counting_seqs: torch.Tensor


class SIGKernel(PointProcessKernel):
    def __init__(
        self,
        static_kernel_type: Literal["linear", "rbf"],
        embedding_type: Literal["linear_interpolant", "constant_interpolant"],
        dyadic_order: int,
        num_event_types: int,
        sigma: Optional[float] = None,
    ):
        match static_kernel_type:
            case "linear":
                self.kernel = SigKernel(
                    static_kernel=LinearKernel(), dyadic_order=dyadic_order
                )
            case "rbf":
                self.kernel = SigKernel(
                    static_kernel=RBFKernel(sigma=sigma), dyadic_order=dyadic_order
                )

        self.embedding_type = embedding_type
        self.num_event_types = num_event_types

    def _get_embedding(
        self, time_seqs: torch.Tensor, type_seqs: torch.Tensor
    ) -> torch.Tensor:
        """Get the embedding of the sequences. Concatenate ponctual point process and time counting process.
        Args:
            time_seqs: Batch of sequences of shape (B, L)
            type_seqs: Batch of type sequences of shape (B, L) with integer type indices
        Returns:
            torch.Tensor: (B, L, 2 + num_event_types) or (B, 2*L, 2 + num_event_types).
        """

        # Ne gère pas encore les masques à implementer !

        match self.embedding_type:
            case "linear_interpolant":
                # Normalize time sequences to [0, 1] range per sequence
                # Add small epsilon to avoid division by zero
                time_max = time_seqs.max(dim=1, keepdim=True)[0] + 1e-8
                normalized_time_seqs = time_seqs / time_max  # (B, L)

                # Normalize counting sequences to [0, 1] range
                counting_seqs = torch.arange(
                    time_seqs.shape[1], device=time_seqs.device, dtype=time_seqs.dtype
                ).expand(time_seqs.shape[0], -1)  # (B, L)
                normalized_counting_seqs = counting_seqs / (
                    time_seqs.shape[1] - 1 + 1e-8
                )  # (B, L)

                # Create one-hot encoding for types
                # type_seqs is (B, L) with integer indices
                type_one_hot = torch.nn.functional.one_hot(
                    type_seqs.long(), num_classes=self.num_event_types
                ).to(time_seqs.dtype)  # (B, L, num_event_types)

                return torch.cat(
                    [
                        normalized_time_seqs.unsqueeze(-1),
                        normalized_counting_seqs.unsqueeze(-1),
                        type_one_hot,
                    ],
                    dim=-1,
                )  # (B, L, 2 + num_event_types)
            case "constant_interpolant":
                # Normalize time sequences to [0, 1] range per sequence
                time_max = time_seqs.max(dim=1, keepdim=True)[0] + 1e-8
                normalized_time_seqs = time_seqs / time_max  # (B, L)

                # create the sequence of [t_1, t_1, t_2, t_2, ..., t_n, t_n]
                pair_indexes = (
                    torch.arange(time_seqs.shape[1], device=time_seqs.device).expand(
                        time_seqs.shape[0], -1
                    )
                    * 2
                )
                out_time_seqs = torch.zeros(
                    (time_seqs.shape[0], time_seqs.shape[1] * 2),
                    dtype=time_seqs.dtype,
                    device=time_seqs.device,
                )  # (B, 2*L)
                out_time_seqs[:, pair_indexes] = normalized_time_seqs
                out_time_seqs[:, pair_indexes + 1] = normalized_time_seqs

                # Normalize counting sequences
                counting_seqs = torch.arange(
                    time_seqs.shape[1], dtype=time_seqs.dtype, device=time_seqs.device
                ).expand(time_seqs.shape[0], -1)
                normalized_counting_seqs = counting_seqs / (
                    time_seqs.shape[1] - 1 + 1e-8
                )

                padded_counting_seqs = torch.zeros(
                    (time_seqs.shape[0], time_seqs.shape[1] * 2),
                    dtype=time_seqs.dtype,
                    device=time_seqs.device,
                )
                padded_counting_seqs[:, pair_indexes] = normalized_counting_seqs
                padded_counting_seqs[:, pair_indexes + 1] = normalized_counting_seqs

                # Create one-hot encoding for types and duplicate
                type_one_hot = torch.nn.functional.one_hot(
                    type_seqs.long(), num_classes=self.num_event_types
                ).to(time_seqs.dtype)  # (B, L, num_event_types)

                # Duplicate one-hot vectors for constant interpolant
                out_type_one_hot = torch.zeros(
                    (time_seqs.shape[0], time_seqs.shape[1] * 2, self.num_event_types),
                    dtype=time_seqs.dtype,
                    device=time_seqs.device,
                )  # (B, 2*L, num_event_types)
                out_type_one_hot[:, pair_indexes] = type_one_hot
                out_type_one_hot[:, pair_indexes + 1] = type_one_hot

                return torch.cat(
                    [
                        out_time_seqs.unsqueeze(-1),
                        padded_counting_seqs.unsqueeze(-1),
                        out_type_one_hot,
                    ],
                    dim=-1,
                )  # (B, 2*L, 2 + num_event_types)
            case _:
                raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    def graam_matrix(
        self,
        phi_batch: Batch | SimulationResult,
        psi_batch: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Compute the Gram matrix between two batches of sequences.
        args:
            phi_time_seqs: (B1, L) L is the number of events in phi
            phi_type_seqs: (B1, L)
            psi_time_seqs: (B2, K) K is the number of events in psi
            psi_type_seqs: (B2, K)
        returns:
            torch.Tensor: (B1, B2)
        """

        phi_time_seqs = phi_batch.time_seqs
        phi_type_seqs = phi_batch.type_seqs
        psi_time_seqs = psi_batch.time_seqs
        psi_type_seqs = psi_batch.type_seqs

        # Store original dtype for conversion back
        original_dtype = phi_time_seqs.dtype

        # Convert to float64 for sigkernel computation
        phi_time_seqs = phi_time_seqs.double()
        phi_type_seqs = phi_type_seqs.double()
        psi_time_seqs = psi_time_seqs.double()
        psi_type_seqs = psi_type_seqs.double()

        # 1) Compute embeddings
        # ---------------------
        # phi : (B1, Lφ, C)
        # psi : (B2, Lψ, C)
        phi_emb = self._get_embedding(phi_time_seqs, phi_type_seqs)
        psi_emb = self._get_embedding(psi_time_seqs, psi_type_seqs)

        # 3) Compute full Gram matrix between all (b,i) and all (b',j)
        # ------------------------------------------------------------
        # output shape from SigKernel = (B1, B2)
        gram: torch.Tensor = self.kernel.compute_Gram(phi_emb, psi_emb)  # type: ignore

        # Convert back to original dtype
        gram = gram.to(original_dtype)

        return gram
