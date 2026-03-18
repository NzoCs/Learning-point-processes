from typing import Literal, cast
from .kernel_protocol import IPointProcessKernel


class PointProcessKernelConfig:
    """Configuration for kernel used in statistical tests.

    Args:
        point_process_kernel_type: Type of point process kernel to use. One of "m_kernel", "sig_kernel".
        space_kernel_type: Type of kernel to use. One of "rbf", "linear".
        embedding_dim: Dimension of the embedding space (only for "embedding" kernel).
        sigma: Bandwidth parameter for RBF kernels (only for "embedding" kernel).
        scaling: Scaling factor for the kernel (only for "embedding" kernel).
    """

    def __init__(
        self,
        point_process_kernel_type: Literal["m_kernel", "sig_kernel"],
        space_kernel_type: Literal["rbf", "linear"],
        embedding_dim: int = 8,
        sigma: float = 1.0,
        scaling: float = 1.0,
        num_discretization_points: int = 100,
        embedding_type: Literal[
            "linear_interpolant", "constant_interpolant"
        ] = "linear_interpolant",
        dyadic_order: int = 0,
    ):
        self.point_process_kernel_type = point_process_kernel_type
        self.space_kernel_type = space_kernel_type
        self.embedding_dim = embedding_dim
        self.sigma = sigma
        self.scaling = scaling
        self.num_discretization_points = num_discretization_points
        self.embedding_type = embedding_type
        self.dyadic_order = dyadic_order

    def create_instance(self, num_classes: int) -> IPointProcessKernel:
        from .space_kernels.config import SpaceKernelConfig

        space_config = SpaceKernelConfig(
            kernel_type=cast(Literal["rbf", "linear"], self.space_kernel_type),
            num_classes=num_classes,
            sigma=self.sigma,
            scaling=self.scaling,
        )
        space_config.point_process_kernel_type = cast(
            Literal["m_kernel", "sig_kernel"], self.point_process_kernel_type
        )
        space_config.embedding_dim = self.embedding_dim

        kernels = space_config.create_instance()

        if self.point_process_kernel_type == "m_kernel" and isinstance(kernels, tuple):
            from .m_kernel import MKernel

            # kernels is a tuple: (time_kernel, type_kernel)
            return MKernel(time_kernel=kernels[0], type_kernel=kernels[1])

        elif self.point_process_kernel_type == "sig_kernel" and not isinstance(
            kernels, tuple
        ):
            from .sig_kernel import SIGKernel

            # kernels is just time_kernel
            return SIGKernel(
                static_kernel=kernels,
                embedding_type=cast(
                    Literal["linear_interpolant", "constant_interpolant"],
                    self.embedding_type,
                ),
                num_discretization_points=self.num_discretization_points,
                dyadic_order=self.dyadic_order,
                num_event_types=num_classes,
            )
        else:
            raise ValueError(
                f"Unknown point process kernel type: {self.point_process_kernel_type}"
            )
