from re import S

import torch
from sigkernel import SigKernel, LinearKernel, RBFKernel
from typing import Optional, TypedDict, Literal

from .kernel_protocol import IPointProcessKernel
from new_ltpp.shared_types import Batch, SimulationResult


class Embedding(TypedDict):
    time_seqs: torch.Tensor
    counting_seqs: torch.Tensor


class SIGKernel(IPointProcessKernel):
    def __init__(
        self,
        static_kernel_type: Literal["linear", "rbf"],
        embedding_type: Literal["linear_interpolant", "constant_interpolant"],
        dyadic_order: int,
        num_event_types: int,
    ):
        self.static_kernel_type = static_kernel_type
        self.embedding_type = embedding_type
        self.dyadic_order = dyadic_order

        # Initialize with a default kernel; will be set properly in compute_gram_matrix based on static_kernel_type
        self.kernel = SigKernel(
            static_kernel=LinearKernel(), dyadic_order=self.dyadic_order
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
                # NOTE: No normalization here - should be done globally in compute_gram_matrix
                # to preserve relative time scales between sequences
                normalized_time_seqs = time_seqs  # (B, L)

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
                # NOTE: No normalization here - should be done globally in compute_gram_matrix
                # to preserve relative time scales between sequences
                normalized_time_seqs = time_seqs  # (B, L)

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

    def _forward_fill_padding(
        self, emb: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Replace padded positions with the last valid embedding value.

        A constant path after the last event produces zero increments,
        so the signature kernel naturally ignores it.

        Args:
            emb: Embedding tensor (B, L', C) where L' may be L or 2*L for constant_interpolant.
            mask: Valid event mask (B, L) with True for real events.

        Returns:
            Embedding with padded positions forward-filled. Same shape as input.
        """
        B, L_emb, C = emb.shape
        B_mask, L_mask = mask.shape

        # For constant_interpolant, L_emb = 2 * L_mask: expand mask to match
        if L_emb == 2 * L_mask:
            # Each event i maps to positions 2*i and 2*i+1
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 2).reshape(B, L_emb)
        else:
            expanded_mask = mask

        # Find the index of the last valid position per sequence
        # last_valid_idx: (B,) — index of the last True in the mask
        last_valid_idx = expanded_mask.long().cumsum(dim=1).argmax(dim=1)  # (B,)

        # Gather the last valid embedding for each sequence
        last_valid_emb = emb[
            torch.arange(B, device=emb.device), last_valid_idx
        ]  # (B, C)

        # Expand to full sequence length and fill where mask is False
        fill_value = last_valid_emb.unsqueeze(1).expand_as(emb)  # (B, L', C)
        emb = torch.where(expanded_mask.unsqueeze(-1), emb, fill_value)

        return emb

    @torch.compile
    def compute_gram_matrix(
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

        phi_time_seqs = phi_time_seqs.double()
        psi_time_seqs = psi_time_seqs.double()

        # Apply uniform normalization across both batches to preserve relative structure
        # Use masked max to ignore padded positions
        phi_masked = phi_time_seqs.masked_fill(~phi_batch.valid_event_mask, 0.0)
        psi_masked = psi_time_seqs.masked_fill(~psi_batch.valid_event_mask, 0.0)
        global_max = max(phi_masked.max().item(), psi_masked.max().item()) + 1e-8
        phi_time_seqs = phi_time_seqs / global_max
        psi_time_seqs = psi_time_seqs / global_max

        match self.static_kernel_type:
            case "linear":
                self.kernel.static_kernel = LinearKernel()
            case "rbf":
                sigma = max(
                    (phi_time_seqs.unsqueeze(1) - psi_time_seqs.unsqueeze(-1))
                    .abs()
                    .median()
                    .item(),
                    1e-8,
                )  # Median heuristic for bandwidth
                self.kernel.static_kernel = RBFKernel(sigma=sigma)  # type: ignore
            case _:
                raise ValueError(
                    f"Unknown static kernel type: {self.static_kernel_type}"
                )

        # 1) Compute embeddings (stays in original dtype, float32 is fine for sigkernel)
        # ---------------------
        # phi : (B1, Lφ, C)
        # psi : (B2, Lψ, C)
        phi_emb = self._get_embedding(phi_time_seqs, phi_type_seqs)
        psi_emb = self._get_embedding(psi_time_seqs, psi_type_seqs)

        # 2) Mask out padded positions by forward-filling with the last valid value.
        #    A constant path has zero increments → no contribution to the signature kernel.
        phi_emb = self._forward_fill_padding(phi_emb, phi_batch.valid_event_mask)
        psi_emb = self._forward_fill_padding(psi_emb, psi_batch.valid_event_mask)

        # 3) Compute full Gram matrix between all (b,i) and all (b',j)
        # ------------------------------------------------------------
        # output shape from SigKernel = (B1, B2)
        gram: torch.Tensor = self.kernel.compute_Gram(phi_emb, psi_emb)  # type: ignore

        return gram
