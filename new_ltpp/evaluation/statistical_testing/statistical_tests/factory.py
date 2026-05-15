from typing import TYPE_CHECKING
from .mmd_test import MMDTwoSampleTest
from .base_test import ITest
from ..point_process_kernels.factory import create_point_process_kernel

if TYPE_CHECKING:
    from new_ltpp.configs.statistical_test_config import StatisticalTestConfig


def create_statistical_test(config: "StatisticalTestConfig") -> ITest:
    """Factory function to create a statistical test based on the config.

    Args:
        config: The statistical test configuration.

    Returns:
        An instance of ITest.
    """
    kernel = create_point_process_kernel(config)

    if config.test_type == "mmd":
        return MMDTwoSampleTest(kernel=kernel, n_samples=config.n_samples)
    elif config.test_type == "ksd":
        raise NotImplementedError("KSD test is not yet implemented.")
    else:
        raise ValueError(f"Unknown statistical test type: {config.test_type}")
