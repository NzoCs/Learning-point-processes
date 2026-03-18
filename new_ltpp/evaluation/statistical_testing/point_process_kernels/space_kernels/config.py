from typing import Literal, Optional
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels.embedding import (
    EmbeddingKernel,
)

from .linear_kernel import (
    LinearKernel,
)
from .rbf import RBFTimeKernel


class SpaceKernelConfig:
    """Configuration for space kernels used in statistical tests. Creates an instance of both time kernel
    and type kernel based on the specified parameters.

    """

    kernel_type: Literal["rbf", "linear"]
    point_process_kernel_type: Literal["m_kernel", "sig_kernel"] | None = None
    num_classes: int

    sigma: Optional[float] = None

    embedding_dim: int = 8
    embedding_sigma: float

    scaling: float = 1.0

    def __init__(
        self,
        kernel_type: Literal["rbf", "linear"],
        num_classes: int,
        sigma: Optional[float] = None,
        scaling: float = 1.0,
    ) -> None:
        """
        Args:
            kernel_type: Type of kernel ('rbf', 'linear', etc.)
            scaling: Scaling factor for the kernel output. Default is 1.0.

        RBF kernel parameters:
            sigma: Bandwidth parameter for RBF kernel. If None, will use median heuristic.

        Linear kernel parameters:
            None.

        Embedding kernel parameters:
            embedding_dim: Dimensionality of the embedding space.
            num_classes: Number of unique event types. Must be specified if kernel_type is 'embedding'.
            embedding_sigma: Optional. Bandwidth parameter for the embedding kernel. If None, will use median heuristic.
        """
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.scaling = scaling
        self.num_classes = num_classes

    def create_instance(
        self,
    ) -> (
        tuple[RBFTimeKernel | LinearKernel, EmbeddingKernel]
        | RBFTimeKernel
        | LinearKernel
    ):
        """Create an instance of the kernel based on the configuration.
        if point_process_kernel_type is 'm_kernel', returns a tuple of (time_kernel, type_kernel).
        if point_process_kernel_type is 'sig_kernel', returns only the time_kernel.

        Args:
            None.
        Returns:
            An instance of the specified kernel(s).
        """

        if self.kernel_type == "rbf":
            if self.sigma is None:
                raise ValueError("Sigma must be specified for RBF kernel")
            time_kernel = RBFTimeKernel(sigma=self.sigma, scaling=self.scaling)

        elif self.kernel_type == "linear":
            time_kernel = LinearKernel(scaling=self.scaling)

        else:
            raise ValueError(
                f"Unknown kernel_type: {self.kernel_type}. "
                f"Supported types: 'rbf', 'linear'"
            )

        type_kernel = EmbeddingKernel(
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
        )

        if self.point_process_kernel_type == "m_kernel":
            return time_kernel, type_kernel
        elif self.point_process_kernel_type == "sig_kernel":
            return time_kernel
        else:
            raise ValueError(
                f"Unknown point_process_kernel_type: {self.point_process_kernel_type}. "
                f"Supported types: 'm_kernel', 'sig_kernel'"
            )
