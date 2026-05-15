from typing import TYPE_CHECKING
from .mmd import MMD
from .base_stat_metric import IStatMetric
from ..point_process_kernels.factory import create_point_process_kernel

if TYPE_CHECKING:
    from new_ltpp.configs.statistical_test_config import StatisticalTestConfig


def create_stat_metric(config: "StatisticalTestConfig") -> IStatMetric:
    """Factory function to create a statistical metric based on the config.

    Args:
        config: The statistical test configuration.

    Returns:
        An instance of IStatMetric.
    """
    kernel = create_point_process_kernel(config)

    if config.test_type == "mmd":
        return MMD(kernel=kernel)
    elif config.test_type == "ksd":
        raise NotImplementedError("KSD metric is not yet implemented.")
    else:
        raise ValueError(f"Unknown statistical test type: {config.test_type}")
