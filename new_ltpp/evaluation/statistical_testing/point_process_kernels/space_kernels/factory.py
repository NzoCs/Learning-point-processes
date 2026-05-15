from typing import TYPE_CHECKING
from .rbf import RBFKernel
from .linear_kernel import LinearKernel
from .protocol import ISpaceKernel

if TYPE_CHECKING:
    from new_ltpp.configs.statistical_test_config import StatisticalTestConfig


def create_space_kernel(config: "StatisticalTestConfig") -> ISpaceKernel:
    """Factory function to create a space kernel based on the config.

    Args:
        config: The statistical test configuration.

    Returns:
        An instance of ISpaceKernel.
    """
    if config.space_kernel_type == "rbf":
        return RBFKernel(sigma=config.sigma, scaling=config.scaling)
    elif config.space_kernel_type == "linear":
        return LinearKernel(scaling=config.scaling)
    else:
        raise ValueError(f"Unknown space kernel type: {config.space_kernel_type}")
