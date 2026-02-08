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
        sigma: float = 1.0,
        transform: MKernelTransform | str = MKernelTransform.EXPONENTIAL,
        **transform_kwargs,
    ):
        """Initialize the M-kernel.

        Args:
            time_kernel: Kernel for time sequences.
            type_kernel: Kernel for type sequences.
            sigma: Bandwidth parameter for the M-kernel.
            transform: Type de transformation à appliquer sur la distance MMD.
            **transform_kwargs: Paramètres additionnels pour la transformation:
                - IMQ: c (default=1.0), beta (default=0.5)
                - RATIONAL_QUADRATIC: alpha (default=1.0)
                - LINEAR: (pas de paramètres supplémentaires)
                - CAUCHY: (pas de paramètres supplémentaires)
        """
        self.time_kernel = time_kernel
        self.type_kernel = type_kernel
        self.sigma = sigma

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
        if self.transform == MKernelTransform.EXPONENTIAL:
            # RBF-like: exp(-d²/(2σ²))
            return torch.exp(-dist_sq / (2 * self.sigma**2))

        elif self.transform == MKernelTransform.IMQ:
            # Inverse Multiquadric: (c² + d²)^(-β)
            return torch.pow(self.c**2 + dist_sq, -self.beta)

        elif self.transform == MKernelTransform.RATIONAL_QUADRATIC:
            # Rational Quadratic: (1 + d²/(2ασ²))^(-α)
            return torch.pow(
                1 + dist_sq / (2 * self.alpha * self.sigma**2), -self.alpha
            )

        elif self.transform == MKernelTransform.LAPLACIAN:
            # Laplacian: exp(-√d²/σ) = exp(-|d|/σ)
            dist = torch.sqrt(dist_sq + 1e-8)
            return torch.exp(-dist / self.sigma)

        elif self.transform == MKernelTransform.LINEAR:
            # Linear: max(0, 1 - d²/σ²)
            return torch.clamp(1 - dist_sq / self.sigma**2, min=0.0)

        elif self.transform == MKernelTransform.CAUCHY:
            # Cauchy: 1/(1 + d²/σ²)
            return 1.0 / (1 + dist_sq / self.sigma**2)

        else:
            raise ValueError(f"Unknown transform: {self.transform}")

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

        # Normalize delta time sequences to [0, 1] range per sequence
        phi_max = phi_delta_time_seqs.max(dim=1, keepdim=True)[0] + 1e-8
        phi_delta_time_seqs = phi_delta_time_seqs / phi_max  # (B1, L)

        psi_max = psi_delta_time_seqs.max(dim=1, keepdim=True)[0] + 1e-8
        psi_delta_time_seqs = psi_delta_time_seqs / psi_max  # (B2, K)

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

        Kt_XX_hat = (Kt_XX_matrix.sum(-1).sum(-1) - Kt_XX_matrix.diagonal().sum(-1)) / (
            L * (L - 1)
        )  # (B1,)
        Kt_YY_hat = (Kt_YY_matrix.sum(-1).sum(-1) - Kt_YY_matrix.diagonal().sum(-1)) / (
            K * (K - 1)
        )  # (B2,)
        Kt_XY_matrix = (Kt_XY_matrix * marks_kernel_matrix).sum(-1).sum(-1) / (
            L * K
        )  # (B1, B2)

        dist_sq = (
            Kt_XX_hat.unsqueeze(-1) + Kt_YY_hat.unsqueeze(0) - 2 * Kt_XY_matrix
        )  # (B1, B2)

        # Applique la transformation choisie pour convertir la distance en kernel
        return self._apply_transform(dist_sq)  # (B1, B2)
