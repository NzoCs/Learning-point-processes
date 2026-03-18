from typing import Any, Dict, List, Literal, Self, TypedDict, cast

from .base_test import ITest
from .configs import TestType, StatisticalTestConfig
from new_ltpp.evaluation.statistical_testing.point_process_kernels import (
    PointProcessKernelConfig,
)
from new_ltpp.utils import logger


class StatisticalTestDict(TypedDict):
    """
    Type mapping for StatisticalTestBuilder configuration state.

    Attributes:
        test_type: The type of statistical test, e.g., 'mmd' or 'ksd'.
        n_permutations: Number of permutations for the MMD permutation test.
        n_samples: Number of samples for KSD estimation.
        point_process_kernel_type: The higher-level kernel operating on point processes ('m_kernel' or 'sig_kernel').
        space_kernel_type: The base spatial/time kernel type to utilize ('rbf' or 'linear').
        embedding_dim: Spatial embedding dimension for categorical event types.
        sigma: Bandwidth parameter for the RBF kernel.
        scaling: General scaling factor applied to the kernel.
        num_discretization_points: Number of discretization points for the signature kernel path integration.
        embedding_type: Path interpolation mode for the signature kernel ('linear_interpolant' or 'constant_interpolant').
        dyadic_order: Dyadic order approximation level for the signature kernel.
        num_classes: Number of unique event types (classes), required to correctly size the embeddings.
    """

    test_type: TestType | None
    n_permutations: int | None
    n_samples: int | None
    point_process_kernel_type: Literal["m_kernel", "sig_kernel"] | None
    space_kernel_type: Literal["rbf", "linear"] | None
    embedding_dim: int | None
    sigma: float | None
    scaling: float | None
    num_discretization_points: int | None
    embedding_type: Literal["linear_interpolant", "constant_interpolant"] | None
    dyadic_order: int | None
    num_classes: int | None


class StatisticalTestBuilder:
    """
    Builder for Statistical Tests.
    Provides a fluent interface to configure and build a statistical test instance
    (e.g., MMD, KSD) along with its underlying point process and space kernels.

    Required Parameters (must be set):
    ===================================
    - set_test_type("mmd" | "ksd"): Type of the statistical test.
    - set_point_process_kernel_type("m_kernel" | "sig_kernel"): Type of point process kernel.
    - set_space_kernel_type("rbf" | "linear"): Base spatial/time kernel to utilize.
    - set_num_classes(int): Number of unique event types (classes), needed for embedding.

    Test-Specific Required Parameters:
    ==================================
    - set_n_permutations(int): Number of permutations for the test. Required if test_type="mmd".
    - set_n_samples(int): Number of samples. Required if test_type="ksd".

    Optional / Kernel-Specific Parameters:
    ======================================
    - set_embedding_dim(int): Spatial embedding dimension for event types (default: 8).
    - set_sigma(float): Bandwidth parameter for the RBF kernel (default: 1.0).
    - set_scaling(float): General scaling factor applied to the kernel (default: 1.0).
    - set_sig_kernel_params(...): Parameters exclusive to the signature kernel (sig_kernel):
        * num_discretization_points (int): Discretization points for path integration (default: 100).
        * embedding_type ("linear_interpolant" | "constant_interpolant"): Path interpolation mode (default: "linear_interpolant").
        * dyadic_order (int): Dyadic order for the signature kernel approximations (default: 0).
    """

    _config_dict: StatisticalTestDict

    def __init__(self):
        self._config_dict = {
            "test_type": None,
            "n_permutations": None,
            "n_samples": None,
            "point_process_kernel_type": None,
            "space_kernel_type": None,
            "embedding_dim": 8,
            "sigma": 1.0,
            "scaling": 1.0,
            "num_discretization_points": 100,
            "embedding_type": "linear_interpolant",
            "dyadic_order": 0,
            "num_classes": None,
        }

    @property
    def config_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self._config_dict)

    def from_dict(self, config_dict: StatisticalTestDict) -> Self:
        """Load configuration from a dictionary matching StatisticalTestDict structure."""
        self._config_dict.update(config_dict)
        return self

    def get_unset_required_fields(self) -> List[str]:
        required = [
            "test_type",
            "point_process_kernel_type",
            "space_kernel_type",
            "num_classes",
        ]

        # Add test-specific required fields
        if self._config_dict.get("test_type") == "mmd":
            required.append("n_permutations")
        elif self._config_dict.get("test_type") == "ksd":
            required.append("n_samples")

        unset = []
        for field in required:
            if self._config_dict.get(field) is None:
                unset.append(field)
        return unset

    def build(self) -> ITest:
        """Create the statistical test instance directly from the configured parameters."""
        unset_fields = self.get_unset_required_fields()
        if len(unset_fields) > 0:
            raise ValueError(
                f"Cannot build StatisticalTest, required fields not set: {unset_fields}"
            )

        logger.info("Building Statistical Test with: %s", self.config_dict)

        # Create the PointProcessKernelConfig
        pp_config = PointProcessKernelConfig(
            point_process_kernel_type=cast(
                Literal["m_kernel", "sig_kernel"],
                self._config_dict["point_process_kernel_type"],
            ),
            space_kernel_type=cast(
                Literal["rbf", "linear"], self._config_dict["space_kernel_type"]
            ),
            embedding_dim=self._config_dict["embedding_dim"],  # type: ignore
            sigma=self._config_dict["sigma"],  # type: ignore
            scaling=self._config_dict["scaling"],  # type: ignore
            num_discretization_points=self._config_dict["num_discretization_points"],  # type: ignore
            embedding_type=cast(
                Literal["linear_interpolant", "constant_interpolant"],
                self._config_dict["embedding_type"],
            ),
            dyadic_order=self._config_dict["dyadic_order"],  # type: ignore
        )

        test_config = StatisticalTestConfig(
            test_type=cast(str, self._config_dict["test_type"]),
            kernel_config=pp_config,
            n_permutations=self._config_dict["n_permutations"],
            n_samples=self._config_dict["n_samples"],
        )

        return test_config.create_instance(
            num_classes=cast(int, self._config_dict["num_classes"])
        )

    def set_test_type(self, test_type: TestType) -> Self:
        self._config_dict["test_type"] = test_type
        return self

    def set_n_permutations(self, n_permutations: int) -> Self:
        self._config_dict["n_permutations"] = n_permutations
        return self

    def set_n_samples(self, n_samples: int) -> Self:
        self._config_dict["n_samples"] = n_samples
        return self

    def set_point_process_kernel_type(
        self, kernel_type: Literal["m_kernel", "sig_kernel"]
    ) -> Self:
        self._config_dict["point_process_kernel_type"] = kernel_type
        return self

    def set_space_kernel_type(self, kernel_type: Literal["rbf", "linear"]) -> Self:
        self._config_dict["space_kernel_type"] = kernel_type
        return self

    def set_embedding_dim(self, embedding_dim: int) -> Self:
        self._config_dict["embedding_dim"] = embedding_dim
        return self

    def set_sigma(self, sigma: float) -> Self:
        self._config_dict["sigma"] = sigma
        return self

    def set_scaling(self, scaling: float) -> Self:
        self._config_dict["scaling"] = scaling
        return self

    def set_sig_kernel_params(
        self,
        num_discretization_points: int = 100,
        embedding_type: Literal[
            "linear_interpolant", "constant_interpolant"
        ] = "linear_interpolant",
        dyadic_order: int = 0,
    ) -> Self:
        """Set specialized parameters for the Signature Kernel."""
        self._config_dict["num_discretization_points"] = num_discretization_points
        self._config_dict["embedding_type"] = embedding_type
        self._config_dict["dyadic_order"] = dyadic_order
        return self

    def set_num_classes(self, num_classes: int) -> Self:
        """Set the number of event types (classes) needed to instantiate the kernels."""
        self._config_dict["num_classes"] = num_classes
        return self
