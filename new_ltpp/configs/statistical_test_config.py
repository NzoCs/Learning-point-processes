from pathlib import Path
from typing import Literal

import torch
from pydantic import ConfigDict, Field, PositiveInt, model_serializer

from new_ltpp.configs.base_config import Config
from new_ltpp.configs.config_utils import extract, load_yaml


class ConditionalEvaluationConfig(Config):
    model_config = ConfigDict(extra="forbid")
    history_events: PositiveInt
    future_events: PositiveInt


class SimulationConfig(Config):
    """Simulation seed and optional conditional history/future evaluation."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    conditional_evaluation: ConditionalEvaluationConfig | None = None

    @model_serializer(mode="wrap")
    def serialize_compatible(self, handler):
        value = handler(self)
        if self.conditional_evaluation is None:
            value.pop("conditional_evaluation", None)
        return value

    def execution_contract(self) -> dict:
        if self.conditional_evaluation is not None:
            return {
                "mode": "conditional_holdout",
                **self.conditional_evaluation.model_dump(),
                "time_origin": "last_history_event",
                "sequence_identity": "source_ids_or_snapshot_row_ids_v1",
                "seed": self.seed,
            }
        return {
            "mode": "fixed_event_count",
            "event_count_source": "input_batch_width",
            "batch_size_source": "data_config.data_loading_specs.batch_size",
            "buffer_size": "input_batch_width + generated_event_count",
            "seed": self.seed,
        }


class StatisticalTestConfig(Config):
    """Configuration complète et validée d'un test statistique.

    Tous les champs sont résolus (pas de None).
    Les champs test-spécifiques non pertinents ont une valeur sentinelle (0).
    """

    # Requis
    test_type: Literal["mmd"]
    point_process_kernel_type: Literal["m_kernel", "sig_kernel"]
    space_kernel_type: Literal["rbf", "linear"]
    num_event_types: int
    n_samples: PositiveInt

    # Optionnels avec defaults
    embedding_dim: int = 8
    sigma: float = 1.0
    scaling: float = 1.0
    num_discretization_points: int = 100
    embedding_type: Literal["counting_grid"] = "counting_grid"
    dyadic_order: int = 0
    signature_backend: Literal["pysiglib"] = "pysiglib"
    signature_max_batch: PositiveInt = 64

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
