from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Self, cast

from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    MMDTwoSampleTest,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels import (
    PointProcessKernelConfig,
)
from new_ltpp.utils import logger

from .base_test import ITest


# ---------------------------------------------------------------------------
# Config — aucun champ Optional / None
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatisticalTestConfig:
    """Configuration complète et validée d'un test statistique.

    Tous les champs sont résolus (pas de None).
    Les champs test-spécifiques non pertinents ont une valeur sentinelle (0).
    """

    # Requis
    test_type: Literal["mmd", "ksd"]
    point_process_kernel_type: Literal["m_kernel", "sig_kernel"]
    space_kernel_type: Literal["rbf", "linear"]
    num_classes: int

    # Test-spécifiques (un seul est utilisé selon test_type)
    n_permutations: int = 0   # actif si test_type="mmd"
    n_samples: int = 0        # actif si test_type="ksd"

    # Optionnels avec defaults
    embedding_dim: int = 8
    sigma: float = 1.0
    scaling: float = 1.0
    num_discretization_points: int = 100
    embedding_type: Literal["linear", "constant"] = "linear"
    dyadic_order: int = 0


# ---------------------------------------------------------------------------
# Builder — porte les Optional, résout vers StatisticalTestConfig
# ---------------------------------------------------------------------------

class StatisticalTestBuilder:
    """Builder fluent pour StatisticalTestConfig.

    Required:
        set_test_type / set_point_process_kernel_type / set_space_kernel_type / set_num_classes

    Test-specific required:
        set_n_permutations  →  si test_type="mmd"
        set_n_samples       →  si test_type="ksd"

    Optional (defaults dans StatisticalTestConfig):
        set_embedding_dim / set_sigma / set_scaling / set_sig_kernel_params
    """

    def __init__(self) -> None:
        self._test_type: Literal["mmd", "ksd"] | None = None
        self._point_process_kernel_type: Literal["m_kernel", "sig_kernel"] | None = None
        self._space_kernel_type: Literal["rbf", "linear"] | None = None
        self._num_classes: int | None = None

        self._n_permutations: int | None = None
        self._n_samples: int | None = None

        self._embedding_dim: int | None = None
        self._sigma: float | None = None
        self._scaling: float | None = None
        self._num_discretization_points: int | None = None
        self._embedding_type: Literal["linear", "constant"] | None = None
        self._dyadic_order: int | None = None

    # --- setters ---

    def set_test_type(self, test_type: Literal["mmd", "ksd"]) -> Self:
        self._test_type = test_type
        return self

    def set_n_permutations(self, n: int) -> Self:
        self._n_permutations = n
        return self

    def set_n_samples(self, n: int) -> Self:
        self._n_samples = n
        return self

    def set_point_process_kernel_type(
        self, kernel_type: Literal["m_kernel", "sig_kernel"]
    ) -> Self:
        self._point_process_kernel_type = kernel_type
        return self

    def set_space_kernel_type(self, kernel_type: Literal["rbf", "linear"]) -> Self:
        self._space_kernel_type = kernel_type
        return self

    def set_num_classes(self, num_classes: int) -> Self:
        self._num_classes = num_classes
        return self

    def set_embedding_dim(self, embedding_dim: int) -> Self:
        self._embedding_dim = embedding_dim
        return self

    def set_sigma(self, sigma: float) -> Self:
        self._sigma = sigma
        return self

    def set_scaling(self, scaling: float) -> Self:
        self._scaling = scaling
        return self

    def set_sig_kernel_params(
        self,
        num_discretization_points: int = 100,
        embedding_type: Literal["linear", "constant"] = "linear",
        dyadic_order: int = 0,
    ) -> Self:
        self._num_discretization_points = num_discretization_points
        self._embedding_type = embedding_type
        self._dyadic_order = dyadic_order
        return self

    def from_dict(self, d: Dict[str, Any]) -> Self:
        for key, value in d.items():
            attr = f"_{key}"
            if hasattr(self, attr):
                setattr(self, attr, value)
        return self

    # --- validation ---

    def _get_unset_required_fields(self) -> List[str]:
        required: Dict[str, Any] = {
            "test_type": self._test_type,
            "point_process_kernel_type": self._point_process_kernel_type,
            "space_kernel_type": self._space_kernel_type,
            "num_classes": self._num_classes,
        }
        if self._test_type == "mmd":
            required["n_permutations"] = self._n_permutations
        elif self._test_type == "ksd":
            required["n_samples"] = self._n_samples

        return [k for k, v in required.items() if v is None]

    # --- build : produit une config sans None ---

    def build_config(self) -> StatisticalTestConfig:
        """Valide et retourne une StatisticalTestConfig entièrement résolue."""
        missing = self._get_unset_required_fields()
        if missing:
            raise ValueError(f"Champs requis non renseignés : {missing}")

        defaults = StatisticalTestConfig(
            test_type=cast(Literal["mmd", "ksd"], self._test_type),
            point_process_kernel_type=cast(
                Literal["m_kernel", "sig_kernel"], self._point_process_kernel_type
            ),
            space_kernel_type=cast(Literal["rbf", "linear"], self._space_kernel_type),
            num_classes=cast(int, self._num_classes),
        )

        return StatisticalTestConfig(
            test_type=defaults.test_type,
            point_process_kernel_type=defaults.point_process_kernel_type,
            space_kernel_type=defaults.space_kernel_type,
            num_classes=defaults.num_classes,
            n_permutations=self._n_permutations if self._n_permutations is not None else defaults.n_permutations,
            n_samples=self._n_samples if self._n_samples is not None else defaults.n_samples,
            embedding_dim=self._embedding_dim if self._embedding_dim is not None else defaults.embedding_dim,
            sigma=self._sigma if self._sigma is not None else defaults.sigma,
            scaling=self._scaling if self._scaling is not None else defaults.scaling,
            num_discretization_points=self._num_discretization_points if self._num_discretization_points is not None else defaults.num_discretization_points,
            embedding_type=self._embedding_type if self._embedding_type is not None else defaults.embedding_type,
            dyadic_order=self._dyadic_order if self._dyadic_order is not None else defaults.dyadic_order,
        )

    def build(self) -> ITest:
        """Build la config puis instancie le test."""
        config = self.build_config()
        logger.info("Building StatisticalTest with: %s", config)

        pp_config = PointProcessKernelConfig(
            point_process_kernel_type=config.point_process_kernel_type,
            space_kernel_type=config.space_kernel_type,
            embedding_dim=config.embedding_dim,
            sigma=config.sigma,
            scaling=config.scaling,
            num_discretization_points=config.num_discretization_points,
            embedding_type=config.embedding_type,
            dyadic_order=config.dyadic_order,
        )
        kernel = pp_config.create_instance(num_classes=config.num_classes)

        match config.test_type:
            case "mmd":
                return MMDTwoSampleTest(kernel=kernel, n_permutations=config.n_permutations)
            case "ksd":
                raise NotImplementedError("KSDTest non implémenté.")
            case _:
                raise ValueError(f"test_type inconnu : {config.test_type}")