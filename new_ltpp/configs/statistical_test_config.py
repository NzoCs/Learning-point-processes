from pathlib import Path
from typing import Literal
from pydantic import PositiveFloat, PositiveInt

from new_ltpp.configs.base_config import Config
from new_ltpp.configs.config_utils import load_yaml, extract


class SimulationConfig(Config):
    time_window: PositiveFloat
    batch_size: PositiveInt
    initial_buffer_size: PositiveInt
    seed: int = 42


class StatisticalTestConfig(Config):
    """Configuration complète et validée d'un test statistique.

    Tous les champs sont résolus (pas de None).
    Les champs test-spécifiques non pertinents ont une valeur sentinelle (0).
    """

    # Requis
    test_type: Literal["mmd", "ksd"]
    point_process_kernel_type: Literal["m_kernel", "sig_kernel"]
    space_kernel_type: Literal["rbf", "linear"]
    num_event_types: int
    n_samples: int

    # Optionnels avec defaults
    embedding_dim: int = 8
    sigma: float = 1.0
    scaling: float = 1.0
    num_discretization_points: int = 100
    embedding_type: Literal["linear", "constant"] = "linear"
    dyadic_order: int = 0

    @classmethod
    def from_yaml_components(
        cls,
        yaml_path: str | Path,
        num_event_types: int,
        config_path: str,
    ) -> "StatisticalTestConfig":
        raw_config = extract(load_yaml(yaml_path), config_path)
        raw_config["num_event_types"] = num_event_types
        return cls(**raw_config)
