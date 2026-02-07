"""
Configuration classes for statistical tests (MMD, KSD, etc.).

This module provides configuration classes for statistical hypothesis tests
used to evaluate temporal point process models.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal


TestType = Literal["mmd", "ksd"]


@dataclass
class KernelConfig:
    """Configuration for kernel used in statistical tests.

    Args:
        kernel_type: Type of kernel ('m_kernel', 'sig_kernel', etc.)
        kernel_params: Parameters specific to the kernel type
    """

    kernel_type: str
    kernel_params: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.kernel_params is None:
            self.kernel_params = {}


@dataclass
class MMDTestConfig:
    """Configuration for MMD two-sample test.

    Args:
        kernel_config: Configuration for the kernel
        n_permutations: Number of permutations for the permutation test
    """

    kernel_config: KernelConfig | Dict[str, Any]
    n_permutations: int = 100

    def __post_init__(self):
        if isinstance(self.kernel_config, dict):
            self.kernel_config = KernelConfig(**self.kernel_config)


@dataclass
class KSDTestConfig:
    """Configuration for Kernelized Stein Discrepancy test.

    Args:
        kernel_config: Configuration for the kernel
        n_samples: Number of samples for KSD estimation
    """

    kernel_config: KernelConfig | Dict[str, Any]
    n_samples: int = 100

    def __post_init__(self):
        if isinstance(self.kernel_config, dict):
            self.kernel_config = KernelConfig(**self.kernel_config)
