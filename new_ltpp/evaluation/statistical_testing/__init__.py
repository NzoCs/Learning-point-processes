from .statistical_metrics import IStatMetric
from .statistical_tests import ITest, MMDTwoSampleTest  # , SteinTest
from .statistical_tests_configs import (
    KernelConfig,
    MMDTestConfig,
    KSDTestConfig,
    TestType,
)
from .statistical_tests_factory import (
    create_kernel_from_config,
    create_mmd_test_from_config,
    # create_ksd_test_from_config,
    create_test_from_config,
)
from .config_builders import (
    KernelConfigBuilder,
    MMDTestConfigBuilder,
    KSDTestConfigBuilder,
)

__all__ = [
    # Protocols
    "IStatMetric",
    "ITest",
    # Tests
    "MMDTwoSampleTest",
    # Configs
    "KernelConfig",
    "MMDTestConfig",
    "KSDTestConfig",
    "TestType",
    # Builders
    "KernelConfigBuilder",
    "MMDTestConfigBuilder",
    "KSDTestConfigBuilder",
    # Factory
    "create_kernel_from_config",
    "create_mmd_test_from_config",
    "create_test_from_config",
]
