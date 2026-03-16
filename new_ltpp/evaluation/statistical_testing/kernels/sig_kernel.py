import torch
from sigkernel import SigKernel
from typing import Optional, TypedDict, Literal

from .kernel_protocol import IPointProcessKernel
from .utils import _get_embedding, _forward_fill_padding
from .space_kernels import LinearTimeKernel, RBFTimeKernel
from new_ltpp.shared_types import Batch, SimulationResult


class Embedding(TypedDict):
    time_seqs: torch.Tensor
    counting_seqs: torch.Tensor


class SIGKernel(IPointProcessKernel):
    def __init__(
        self,
        static_kernel: ISpaceKernel,
        embedding_type: Literal["linear_interpolant", "constant_interpolant"],
        num_discretization_points: int,
        dyadic_order: int,
        num_event_types: int,
    ):
        self.embedding_type = embedding_type
        self.num_discretization_points = num_discretization_points
        self.dyadic_order = dyadic_order

        # Initialize with a default kernel; will be set properly in compute_gram_matrix based on static_kernel_type
        self.kernel = SigKernel(
            static_kernel=static_kernel, dyadic_order=self.dyadic_order
        )

        self.embedding_type = embedding_type
        self.num_event_types = num_event_types

    def _prepare_kernel(
        self, phi_batch: Batch | SimulationResult, psi_batch: Batch | SimulationResult
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the SigKernel instance with the appropriate static kernel based on the embedding."""

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
        global_max = torch.max(phi_masked.max(), psi_masked.max()) + 1e-8
        phi_time_seqs = phi_time_seqs / global_max
        psi_time_seqs = psi_time_seqs / global_max

        match self.static_kernel_type:
            case "linear":
                self.kernel.static_kernel = LinearKernel()
            case "rbf":
                sigma = torch.max(
                    (phi_time_seqs.unsqueeze(1) - psi_time_seqs.unsqueeze(-1))
                    .abs()
                    .median(),
                    torch.tensor(1e-8, device=phi_time_seqs.device),
                )  # Median heuristic for bandwidth
                if self.rbf_scaling is not None:
                self.kernel.static_kernel = RBFKernel(sigma=sigma)  # type: ignore
            case _:
                raise ValueError(
                    f"Unknown static kernel type: {self.static_kernel_type}"
                )

        # 1) Compute embeddings (stays in original dtype, float32 is fine for sigkernel)
        # ---------------------
        # phi : (B, Lφ, C)
        # psi : (B, Lψ, C)
        phi_emb = _get_embedding(
            self.num_discretization_points,
            self.embedding_type,
            self.num_event_types,
            phi_time_seqs,
            phi_type_seqs,
        )
        psi_emb = _get_embedding(
            self.num_discretization_points,
            self.embedding_type,
            self.num_event_types,
            psi_time_seqs,
            psi_type_seqs,
        )

        # 2) Mask out padded positions by forward-filling with the last valid value.
        #    A constant path has zero increments → no contribution to the signature kernel.
        phi_emb = _forward_fill_padding(phi_emb, phi_batch.valid_event_mask)
        psi_emb = _forward_fill_padding(psi_emb, psi_batch.valid_event_mask)

        return phi_emb, psi_emb

    @torch.compile
    def compute_gram_matrix(
        self,
        phi_batch: Batch | SimulationResult,
        psi_batch: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Compute the Gram matrix between two batches of sequences.
        args:
            phi_time_seqs: (B, L) L is the number of events in phi
            phi_type_seqs: (B, L)
            psi_time_seqs: (B, K) K is the number of events in psi
            psi_type_seqs: (B, K)
        returns:
            torch.Tensor: (B, B)
        """
        phi_emb, psi_emb = self._prepare_kernel(phi_batch, psi_batch)

        # 3) Compute full Gram matrix between all (b,i) and all (b',j)
        # ------------------------------------------------------------
        # output shape from SigKernel = (B, B)
        gram: torch.Tensor = self.kernel.compute_Gram(phi_emb, psi_emb)  # type: ignore

        return gram

    def compute_mmd(
        self, phi: Batch | SimulationResult, psi: Batch | SimulationResult
    ) -> torch.Tensor:
        """Compute the MMD distance between two batches of sequences."""
        phi_emb, psi_emb = self._prepare_kernel(phi, psi)

        mmd_dist = self.kernel.compute_mmd(phi_emb, psi_emb)  # type: ignore
        return mmd_dist
