from typing import TYPE_CHECKING
from .m_kernel import MKernel
from .sig_kernel import SIGKernel
from .kernel_protocol import IPointProcessKernel
from .space_kernels.factory import create_space_kernel
from .space_kernels.embedding import EmbeddingKernel

if TYPE_CHECKING:
    from new_ltpp.configs.statistical_test_config import StatisticalTestConfig


def create_point_process_kernel(config: "StatisticalTestConfig") -> IPointProcessKernel:
    """Factory function to create a point process kernel based on the config.

    Args:
        config: The statistical test configuration.

    Returns:
        An instance of IPointProcessKernel.
    """
    space_kernel = create_space_kernel(config)

    if config.point_process_kernel_type == "m_kernel":
        # For M-Kernel, we use the space kernel for time and an EmbeddingKernel for types
        type_kernel = EmbeddingKernel(
            num_classes=config.num_event_types,
            embedding_dim=config.embedding_dim,
        )
        return MKernel(
            time_kernel=space_kernel,
            type_kernel=type_kernel,
            num_points=config.num_discretization_points,
        )
    elif config.point_process_kernel_type == "sig_kernel":
        return SIGKernel(
            static_kernel=space_kernel,
            embedding_type=config.embedding_type,
            num_discretization_points=config.num_discretization_points,
            dyadic_order=config.dyadic_order,
            num_event_types=config.num_event_types,
        )
    else:
        raise ValueError(
            f"Unknown point process kernel type: {config.point_process_kernel_type}"
        )
