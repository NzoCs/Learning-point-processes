import torch
from typing import Literal
from enum import Enum

from .kernel_protocol import IPointProcessKernel
from .space_kernels import ISpaceKernel
from new_ltpp.shared_types import Batch, SimulationResult

LossEnum = Literal["energy", "sinkhorn", "hausdorff", "gaussian", "laplacian"]


class MKernelTransform(Enum):
    """Fonction de transformation pour convertir la distance MMD en kernel."""

    EXPONENTIAL = "exponential"  # exp(-d²/(2σ²)) - RBF-like (default)
    IMQ = "imq"  # (c² + d²)^(-β) - Inverse Multiquadric
    RATIONAL_QUADRATIC = "rq"  # (1 + d²/(2ασ²))^(-α)
    LAPLACIAN = "laplacian"  # exp(-√d²/σ)
    LINEAR = "linear"  # max(0, 1 - d²/σ²)
    CAUCHY = "cauchy"  # 1/(1 + d²/σ²)


TimeKernel = ISpaceKernel
TypeKernel = ISpaceKernel


class MKernel(IPointProcessKernel):
    def __init__(
        self,
        time_kernel: TimeKernel,
        type_kernel: TypeKernel,
        transform: MKernelTransform | str = MKernelTransform.EXPONENTIAL,
        **transform_kwargs,
    ):
        """Initialize the M-kernel.

        Args:
            time_kernel: Kernel for time sequences.
            type_kernel: Kernel for type sequences.
            transform: Type de transformation à appliquer sur la distance MMD.
            **transform_kwargs: Paramètres additionnels pour la transformation:
                - IMQ: c (default=1.0), beta (default=0.5)
                - RATIONAL_QUADRATIC: alpha (default=1.0)
                - LINEAR: (pas de paramètres supplémentaires)
                - CAUCHY: (pas de paramètres supplémentaires)
        """
        self.time_kernel = time_kernel
        self.type_kernel = type_kernel

        if isinstance(transform, str):
            transform = MKernelTransform(transform)
        self.transform = transform

        # Paramètres spécifiques aux transformations
        self.c = transform_kwargs.get("c", 1.0)
        self.beta = transform_kwargs.get("beta", 0.5)
        self.alpha = transform_kwargs.get("alpha", 1.0)

    def _apply_transform(self, dist_sq: torch.Tensor) -> torch.Tensor:
        """Applique la transformation choisie sur la distance au carré.

        Args:
            dist_sq: Distance au carré (B1, B2)

        Returns:
            Kernel value (B1, B2)
        """
        # Ensure non-negative and clamp to prevent NaN
        dist_sq = dist_sq.clamp(min=0.0)

        # Robust median heuristic: filter out zeros and NaNs
        valid_dist = dist_sq[torch.isfinite(dist_sq) & (dist_sq > 1e-10)]
        sigma = valid_dist.median()
        sigma = torch.max(
            sigma, torch.tensor(1e-6, device=dist_sq.device)
        )  # Ensure positive

        if self.transform == MKernelTransform.EXPONENTIAL:
            # RBF-like: exp(-d²/(2σ²))
            return torch.exp(-dist_sq / (2 * sigma))

        elif self.transform == MKernelTransform.IMQ:
            # Inverse Multiquadric: (c² + d²)^(-β)
            return torch.pow(self.c**2 + dist_sq / sigma, -self.beta)

        elif self.transform == MKernelTransform.RATIONAL_QUADRATIC:
            # Rational Quadratic: (1 + d²/(2ασ²))^(-α)
            return torch.pow(1 + dist_sq / (2 * self.alpha * sigma), -self.alpha)

        elif self.transform == MKernelTransform.LAPLACIAN:
            # Laplacian: exp(-√d²/σ) = exp(-|d|/σ)
            dist = torch.sqrt(dist_sq + 1e-8)
            return torch.exp(-dist / sigma)

        elif self.transform == MKernelTransform.LINEAR:
            # Linear: max(0, 1 - d²/σ²)
            return torch.clamp(1 - dist_sq / sigma, min=0.0)

        elif self.transform == MKernelTransform.CAUCHY:
            # Cauchy: 1/(1 + d²/σ²)
            return 1.0 / (1 + dist_sq / sigma)

        else:
            raise ValueError(f"Unknown transform: {self.transform}")

    @torch.compile
    def compute_gram_matrix(
        self,
        phi_batch: Batch | SimulationResult,
        psi_batch: Batch | SimulationResult,
    ) -> torch.Tensor:
        """Compute the M-kernel between two batches of sequences.
        args:
            phi_delta_time_seqs: (B1, L) is the number of events in phi
            phi_type_seqs: (B1, L)
            psi_delta_time_seqs: (B2, K) K is the number of events in psi
            psi_type_seqs: (B2, K)
        returns:
            torch.Tensor: (B1, B2)
        """

        phi_delta_time_seqs = phi_batch.time_delta_seqs
        phi_type_seqs = phi_batch.type_seqs
        psi_delta_time_seqs = psi_batch.time_delta_seqs
        psi_type_seqs = psi_batch.type_seqs

        # Extract valid event masks (True = real event, False = padding)
        phi_mask = phi_batch.valid_event_mask  # (B1, L)
        psi_mask = psi_batch.valid_event_mask  # (B2, K)

        # Per-sequence valid counts
        phi_n = phi_mask.sum(dim=1).float()  # (B1,)
        psi_n = psi_mask.sum(dim=1).float()  # (B2,)

        # Apply uniform normalization across both batches to preserve relative structure
        # Use masked max to ignore padded positions
        phi_masked_dt = phi_delta_time_seqs.masked_fill(~phi_mask, 0.0)
        psi_masked_dt = psi_delta_time_seqs.masked_fill(~psi_mask, 0.0)
        global_max = max(phi_masked_dt.max(), psi_masked_dt.max()) + 1e-8
        phi_delta_time_seqs = phi_delta_time_seqs / global_max
        psi_delta_time_seqs = psi_delta_time_seqs / global_max

        B1, L = phi_delta_time_seqs.shape
        B2, K = psi_delta_time_seqs.shape

        Kt_XX_matrix = self.time_kernel.intra_batch_kernel_matrix(
            phi_delta_time_seqs,
        )  # (B1, L, L)

        Kt_XY_matrix = self.time_kernel.cross_batch_kernel_matrix(
            phi_delta_time_seqs,
            psi_delta_time_seqs,
        )  # (B1, B2, L, K)

        marks_kernel_matrix: torch.Tensor = self.type_kernel.cross_batch_kernel_matrix(
            phi_type_seqs,
            psi_type_seqs,
        )  # (B1, B2, L, K)

        Kt_YY_matrix = self.time_kernel.intra_batch_kernel_matrix(
            psi_delta_time_seqs,
        )  # (B2, K, K)

        # --- Mask out padded positions before summing ---

        # Intra-batch XX: mask shape (B1, L, L) — both positions must be valid
        xx_mask = phi_mask.unsqueeze(-1) & phi_mask.unsqueeze(-2)  # (B1, L, L)
        Kt_XX_masked = Kt_XX_matrix * xx_mask.float()
        # Remove diagonal (self-pairs), then normalize by n_i * (n_i - 1)
        # Clamp denominator to avoid division by zero
        denom_xx = (phi_n * (phi_n - 1)).clamp(min=1e-8)
        Kt_XX_hat = (
            Kt_XX_masked.sum(-1).sum(-1)
            - Kt_XX_masked.diagonal(dim1=-2, dim2=-1).sum(-1)
        ) / denom_xx  # (B1,)

        # Intra-batch YY: mask shape (B2, K, K)
        yy_mask = psi_mask.unsqueeze(-1) & psi_mask.unsqueeze(-2)  # (B2, K, K)
        Kt_YY_masked = Kt_YY_matrix * yy_mask.float()
        # Clamp denominator to avoid division by zero
        denom_yy = (psi_n * (psi_n - 1)).clamp(min=1e-8)
        Kt_YY_hat = (
            Kt_YY_masked.sum(-1).sum(-1)
            - Kt_YY_masked.diagonal(dim1=-2, dim2=-1).sum(-1)
        ) / denom_yy  # (B2,)

        # Cross-batch XY: mask shape (B1, B2, L, K) — phi position i and psi position j both valid
        xy_mask = phi_mask.unsqueeze(1).unsqueeze(-1) & psi_mask.unsqueeze(0).unsqueeze(
            -2
        )  # (B1, B2, L, K)
        Kt_XY_masked = (Kt_XY_matrix * marks_kernel_matrix) * xy_mask.float()
        # Normalize by n_phi_i * n_psi_j for each (i, j) pair
        xy_norm = phi_n.unsqueeze(-1) * psi_n.unsqueeze(0)  # (B1, B2)
        Kt_XY_matrix = Kt_XY_masked.sum(-1).sum(-1) / xy_norm.clamp(min=1)  # (B1, B2)

        dist_sq = (
            Kt_XX_hat.unsqueeze(-1) + Kt_YY_hat.unsqueeze(0) - 2 * Kt_XY_matrix
        )  # (B1, B2)

        # Applique la transformation choisie pour convertir la distance en kernel
        return self._apply_transform(dist_sq)  # (B1, B2)
