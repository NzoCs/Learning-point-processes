"""Builders for statistical test configurations."""

from typing import Any, Dict, List
from typing import Self

from new_ltpp.evaluation.statistical_testing.statistical_tests_configs import (
    KernelConfig,
    MMDTestConfig,
    KSDTestConfig,
)
from new_ltpp.utils import logger


class KernelConfigBuilder:
    """Builder for kernel configurations.

    Usage:
        builder = KernelConfigBuilder()
        builder.set_kernel_type("m_kernel")
            .set_kernel_params({"time_kernel": "rbf", "sigma": 1.0})
        kernel_config = builder.build()
    """

    def __init__(self):
        self._config_dict: Dict[str, Any] = {
            "kernel_type": None,
            "kernel_params": None,
        }

    def set_kernel_type(self, kernel_type: str) -> Self:
        """Set the kernel type.

        Args:
            kernel_type: Type of kernel ('m_kernel', 'sig_kernel')
        """
        self._config_dict["kernel_type"] = kernel_type
        return self

    def set_kernel_params(self, kernel_params: Dict[str, Any]) -> Self:
        """Set kernel-specific parameters.

        Args:
            kernel_params: Dictionary of parameters for the kernel
        """
        self._config_dict["kernel_params"] = kernel_params
        return self

    def build(self) -> KernelConfig:
        """Build the KernelConfig instance."""
        if self._config_dict["kernel_type"] is None:
            raise ValueError("kernel_type must be set")

        return KernelConfig(
            kernel_type=self._config_dict["kernel_type"],
            kernel_params=self._config_dict["kernel_params"] or {},
        )

    def from_dict(self, config_dict: Dict[str, Any]) -> Self:
        """Load configuration from dictionary."""
        self._config_dict = config_dict
        return self


class MMDTestConfigBuilder:
    """Builder for MMD test configurations.

    Usage:
        builder = MMDTestConfigBuilder()
        builder.set_kernel_config(kernel_config)
            .set_n_permutations(200)
        mmd_config = builder.build()
    """

    def __init__(self):
        self._config_dict: Dict[str, Any] = {
            "kernel_config": None,
            "n_permutations": None,  # Default value
        }

    @property
    def required_fields(self) -> List[str]:
        return ["kernel_config"]

    def set_kernel_config(self, kernel_config: KernelConfig) -> Self:
        """Set the kernel configuration.

        Args:
            kernel_config: A KernelConfig instance
        """
        self._config_dict["kernel_config"] = kernel_config
        return self

    def set_n_permutations(self, n_permutations: int) -> Self:
        """Set the number of permutations.

        Args:
            n_permutations: Number of permutations for the test
        """
        self._config_dict["n_permutations"] = n_permutations
        return self

    def build(self) -> MMDTestConfig:
        """Build the MMDTestConfig instance."""
        if self._config_dict["kernel_config"] is None:
            raise ValueError("kernel_config must be set")

        logger.info(
            f"Building MMDTestConfig with n_permutations={self._config_dict['n_permutations']}"
        )

        return MMDTestConfig(
            kernel_config=self._config_dict["kernel_config"],
            n_permutations=self._config_dict["n_permutations"],
        )

    def from_dict(self, config_dict: Dict[str, Any]) -> Self:
        """Load configuration from dictionary."""
        self._config_dict = config_dict
        return self


class KSDTestConfigBuilder:
    """Builder for KSD test configurations.

    Usage:
        builder = KSDTestConfigBuilder()
        builder.set_kernel_config(kernel_config)
            .set_n_samples(100)
        ksd_config = builder.build()
    """

    def __init__(self):
        self._config_dict: Dict[str, Any] = {
            "kernel_config": None,
            "n_samples": None,  # Default value
        }

    @property
    def required_fields(self) -> List[str]:
        return ["kernel_config"]

    def set_kernel_config(self, kernel_config: KernelConfig) -> Self:
        """Set the kernel configuration.

        Args:
            kernel_config: A KernelConfig instance
        """
        self._config_dict["kernel_config"] = kernel_config
        return self

    def set_n_samples(self, n_samples: int) -> Self:
        """Set the number of samples.

        Args:
            n_samples: Number of samples for KSD estimation
        """
        self._config_dict["n_samples"] = n_samples
        return self

    def build(self) -> KSDTestConfig:
        """Build the KSDTestConfig instance."""
        if self._config_dict["kernel_config"] is None:
            raise ValueError("kernel_config must be set")

        logger.info(
            f"Building KSDTestConfig with n_samples={self._config_dict['n_samples']}"
        )

        return KSDTestConfig(
            kernel_config=self._config_dict["kernel_config"],
            n_samples=self._config_dict["n_samples"],
        )

    def from_dict(self, config_dict: Dict[str, Any]) -> Self:
        """Load configuration from dictionary."""
        self._config_dict = config_dict
        return self
