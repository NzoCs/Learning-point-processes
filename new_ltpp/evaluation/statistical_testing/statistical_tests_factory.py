"""Factory for creating statistical test instances from configurations."""

from new_ltpp.evaluation.statistical_testing.statistical_tests_configs import (
    MMDTestConfig,
    KernelConfig,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    MMDTwoSampleTest,
    ITest,
)
from new_ltpp.evaluation.statistical_testing.kernels import (
    MKernel,
    SIGKernel,
    IPointProcessKernel,
    create_time_kernel,
    EmbeddingKernel,
)


def create_kernel_from_config(kernel_config: KernelConfig) -> IPointProcessKernel:
    """Create a kernel instance from a KernelConfig.

    Args:
        kernel_config: Kernel configuration

    Returns:
        Instantiated kernel

    Raises:
        ValueError: If kernel_type is not recognized
    """
    kernel_type = kernel_config.kernel_type.lower()
    params = kernel_config.kernel_params or {}

    if kernel_type == "m_kernel":
        # M-Kernel requires both time_kernel and type_kernel
        time_kernel_type = params.get("time_kernel", "rbf")
        time_kernel_params = params.get("time_kernel_params", {})

        time_kernel = create_time_kernel(
            kernel_type=time_kernel_type, **time_kernel_params
        )

        # Type kernel configuration
        type_kernel_params = params.get("type_kernel_params", {})
        num_classes = type_kernel_params.get("num_classes", 10)
        embedding_dim = type_kernel_params.get("embedding_dim", 32)
        type_kernel_sigma = type_kernel_params.get("sigma", 1.0)

        type_kernel = EmbeddingKernel(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            sigma=type_kernel_sigma,
        )

        # M-Kernel transform parameters
        m_kernel_sigma = params.get("sigma", 1.0)
        m_kernel_transform = params.get("transform", "exponential")

        return MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=m_kernel_sigma,
            transform=m_kernel_transform,
        )

    elif kernel_type == "sig_kernel":
        # SIG-Kernel (signature kernel)
        return SIGKernel(**params)

    else:
        raise ValueError(
            f"Unknown kernel_type: {kernel_type}. "
            f"Supported types: 'm_kernel', 'sig_kernel'"
        )


def create_mmd_test_from_config(config: MMDTestConfig) -> MMDTwoSampleTest:
    """Create an MMD test instance from configuration.

    Args:
        config: MMD test configuration

    Returns:
        Instantiated MMDTwoSampleTest
    """
    kernel = create_kernel_from_config(config.kernel_config)

    return MMDTwoSampleTest(
        kernel=kernel,
        n_permutations=config.n_permutations,
    )


# def create_ksd_test_from_config(config: KSDTestConfig) -> SteinTest:  # noqa: F821
#     """Create a KSD test instance from configuration.

#     Args:
#         config: KSD test configuration

#     Returns:
#         Instantiated SteinTest
#     """
#     kernel = create_kernel_from_config(config.kernel_config)

#     return SteinTest(kernel=kernel)


def create_test_from_config(
    config: MMDTestConfig,
) -> ITest:
    """Create a test instance from configuration (auto-detects type).

    Args:
        config: Test configuration (MMDTestConfig or KSDTestConfig)

    Returns:
        Instantiated test

    Raises:
        ValueError: If config type is not recognized
    """
    if isinstance(config, MMDTestConfig):
        return create_mmd_test_from_config(config)
    # elif isinstance(config, KSDTestConfig):
    #     return create_ksd_test_from_config(config)
    else:
        raise ValueError(
            f"Unknown config type: {type(config).__name__}. "
            f"Expected MMDTestConfig or KSDTestConfig"
        )
