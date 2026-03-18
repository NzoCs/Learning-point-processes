"""
Configuration classes for statistical tests (MMD, KSD, etc.).

This module provides configuration classes for statistical hypothesis tests
used to evaluate temporal point process models.
"""

from typing import Any, Dict, Literal, Optional

from .base_test import ITest

from .mmd_test import MMDTwoSampleTest
from new_ltpp.evaluation.statistical_testing.point_process_kernels import (
    PointProcessKernelConfig,
)


TestType = Literal["mmd", "ksd"]


class StatisticalTestConfig:
    def __init__(
        self,
        test_type: str,
        kernel_config: PointProcessKernelConfig | Dict[str, Any],
        n_permutations: Optional[int] = None,
        n_samples: Optional[int] = None,
        **kwargs: Any,
    ):
        self.test_type = test_type
        if isinstance(kernel_config, dict):
            self.kernel_config = PointProcessKernelConfig(**kernel_config)
        else:
            self.kernel_config = kernel_config
        self.test_params = kwargs
        if n_permutations is not None:
            self.test_params["n_permutations"] = n_permutations
        if n_samples is not None:
            self.test_params["n_samples"] = n_samples

    def create_instance(self, num_classes: int) -> ITest:
        """Create an instance of the statistical test based on the configuration."""
        if self.test_type == "mmd":
            if self.test_params.get("n_permutations") is None:
                raise ValueError("n_permutations must be specified for MMD test")

            return MMDTwoSampleTest(
                kernel=self.kernel_config.create_instance(num_classes=num_classes),
                n_permutations=self.test_params["n_permutations"],
            )
        elif self.test_type == "ksd":
            if self.test_params.get("n_samples") is None:
                raise ValueError("n_samples must be specified for KSD test")

            raise NotImplementedError(
                "KSDTest implementation is not currently provided in locals."
            )

        else:
            raise ValueError(f"Unsupported test type: {self.test_type}")
