import sys
import os

sys.path.append(os.getcwd())


def test_imports():
    print("Testing imports...")
    try:
        from new_ltpp.evaluation.statistical_testing.kernels import (
            MKernel,
            create_time_kernel,
            TimeKernelType,
        )

        print("✅ new_ltpp.evaluation.statistical_testing.kernels imported")
    except ImportError as e:
        print(f"❌ Failed to import kernels: {e}")
        raise

    try:
        from new_ltpp.evaluation.statistical_testing.statistical_tests_configs import (
            MMDTestConfig,
        )

        print(
            "✅ new_ltpp.evaluation.statistical_testing.statistical_tests_configs imported"
        )
    except ImportError as e:
        print(f"❌ Failed to import statistical_tests_configs: {e}")
        raise

    try:
        from new_ltpp.evaluation.statistical_testing.config_builders import (
            MMDTestConfigBuilder,
        )

        print("✅ new_ltpp.evaluation.statistical_testing.config_builders imported")
    except ImportError as e:
        print(f"❌ Failed to import config_builders: {e}")
        raise

    try:
        from new_ltpp.evaluation.statistical_testing.statistical_tests_factory import (
            create_mmd_test_from_config,
        )

        print(
            "✅ new_ltpp.evaluation.statistical_testing.statistical_tests_factory imported"
        )
    except ImportError as e:
        print(f"❌ Failed to import statistical_tests_factory: {e}")
        raise

    # Check graam_matrix rename
    try:
        from new_ltpp.evaluation.statistical_testing.kernels.kernel_protocol import (
            IPointProcessKernel,
        )

        # Check if compute_gram_matrix exists in protocol (abstract method)
        # It won't be callable on Protocol directly, but we can check attributes if runtime_checkable
        print("✅ IPointProcessKernel imported")
    except ImportError as e:
        print(f"❌ Failed to import IPointProcessKernel: {e}")
        raise

    try:
        from new_ltpp.evaluation.statistical_testing.kernels.m_kernel import MKernel

        if hasattr(MKernel, "compute_gram_matrix"):
            print("✅ MKernel.compute_gram_matrix exists")
        else:
            print("❌ MKernel.compute_gram_matrix MISSING")
            raise AttributeError("MKernel missing compute_gram_matrix")

        if hasattr(MKernel, "graam_matrix"):
            print("⚠️ MKernel.graam_matrix still exists (deprecated?)")
        else:
            print("✅ MKernel.graam_matrix removed")

    except ImportError as e:
        print(f"❌ Failed to import MKernel: {e}")
        raise


if __name__ == "__main__":
    try:
        test_imports()
        print("\nAll API import tests passed!")
    except Exception as e:
        print(f"\nAPI import tests FAILED: {e}")
        sys.exit(1)
