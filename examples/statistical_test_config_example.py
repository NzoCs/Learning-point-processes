"""
Example usage of statistical test configurations and factory.

This script demonstrates how to:
1. Create test configurations using builders or dictionaries
2. Instantiate tests using the factory
3. Use the tests to compute p-values
"""

from new_ltpp.evaluation.statistical_testing import (
    # Configs
    KernelConfig,
    MMDTestConfig,
    KernelConfigBuilder,
    MMDTestConfigBuilder,
    KSDTestConfigBuilder,
    # Factory
    create_test_from_config,
)


def example_with_builders():
    """Example using config builders."""
    print("=== Example with Builders ===\n")

    # Create kernel config using builder
    kernel_config = (
        KernelConfigBuilder()
        .set_kernel_type("m_kernel")
        .set_kernel_params(
            {
                "time_kernel": "rbf",
                "time_kernel_params": {"sigma": 1.0},
                "type_kernel_params": {
                    "num_classes": 10,
                    "embedding_dim": 32,
                    "sigma": 1.0,
                },
                "sigma": 1.0,  # M-kernel sigma
                "transform": "exponential",
            }
        )
        .build()
    )

    print(f"Kernel config: {kernel_config}\n")

    # Create MMD test config using builder
    mmd_config = (
        MMDTestConfigBuilder()
        .set_kernel_config(kernel_config)
        .set_n_permutations(200)
        .build()
    )

    print(f"MMD config: {mmd_config}\n")

    # Create test instance from config
    mmd_test = create_test_from_config(mmd_config)
    print(f"MMD test: {mmd_test.name}")
    print(f"  - n_permutations: {mmd_test.n_permutations}")
    print(f"  - kernel: {type(mmd_test.kernel).__name__}\n")


def example_with_dicts():
    """Example using dictionaries directly."""
    print("=== Example with Dictionaries ===\n")

    # Create configs directly from dictionaries
    mmd_config = MMDTestConfig(
        kernel_config={
            "kernel_type": "m_kernel",
            "kernel_params": {
                "time_kernel": "rbf",
                "time_kernel_params": {"sigma": 2.0},
                "type_kernel_params": {
                    "num_classes": 5,
                    "embedding_dim": 16,
                    "sigma": 0.5,
                },
            },
        },
        n_permutations=100,
    )

    print(f"MMD config: {mmd_config}\n")

    # Create test instance
    mmd_test = create_test_from_config(mmd_config)
    print(f"MMD test: {mmd_test.name}")
    print(f"  - n_permutations: {mmd_test.n_permutations}\n")


def example_ksd_test():
    """Example with KSD test."""
    print("=== Example with KSD Test ===\n")

    # Create KSD config using builder
    kernel_config = KernelConfig(kernel_type="sig_kernel", kernel_params={})

    ksd_config = (
        KSDTestConfigBuilder()
        .set_kernel_config(kernel_config)
        .set_n_samples(150)
        .build()
    )

    print(f"KSD config: {ksd_config}\n")

    # Create test instance
    ksd_test = create_test_from_config(ksd_config)
    print(f"KSD test: {ksd_test.name}")
    print(f"  - kernel: {type(ksd_test.kernel).__name__}\n")


def main():
    """Run all examples."""
    example_with_builders()
    print("-" * 50 + "\n")

    example_with_dicts()
    print("-" * 50 + "\n")

    example_ksd_test()


if __name__ == "__main__":
    main()
