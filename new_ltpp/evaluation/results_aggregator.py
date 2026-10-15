from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from new_ltpp.evaluation.accumulators import FinalResult
from new_ltpp.evaluation.statistical_testing.point_process_kernels.utils import (
    SIGNATURE_PATH_PREPARATION,
)
from new_ltpp.utils import logger


class ResultsAggregator:
    """
    Aggregates results from different phases (test, simulation) and metadata
    into a single centralized CSV file.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def add_result(
        self,
        experiment_id: str,
        metadata: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None,
        sim_metrics: Optional[FinalResult] = None,
    ) -> None:
        """
        Add or update a row in the global results CSV.
        """
        # Flatten all information into a single row
        row: Dict[str, Any] = {"experiment_id": experiment_id}

        # Add metadata (prefixed)
        for k, v in metadata.items():
            row[f"meta_{k}"] = v

        # Add test metrics (prefixed)
        if test_metrics:
            for k, v in test_metrics.items():
                row[f"test_{k}"] = v

        # Add sim metrics (prefixed)
        if sim_metrics:
            # If it's a FinalResult, extract the 'metrics' part and 'batch_count'
            metrics_to_add = {}
            if "metrics" in sim_metrics and isinstance(sim_metrics["metrics"], dict):
                metrics_to_add.update(sim_metrics["metrics"])

            if "batch_count" in sim_metrics:
                metrics_to_add["batch_count"] = sim_metrics["batch_count"]

            # If metrics_to_add is empty, it means sim_metrics might already be the dict of scalars
            if not metrics_to_add:
                metrics_to_add = {
                    k: v
                    for k, v in sim_metrics.items()
                    if isinstance(v, (int, float, str, bool))
                }

            for k, v in metrics_to_add.items():
                row[f"sim_{k}"] = v

        new_df = pd.DataFrame([row])

        # Test and simulation rows have different columns. Appending their
        # raw values under the first header produces a malformed CSV.
        if self.csv_path.exists():
            previous = pd.read_csv(self.csv_path)
            new_df = pd.concat([previous, new_df], ignore_index=True, sort=False)
        temporary = self.csv_path.with_suffix(".csv.tmp")
        new_df.to_csv(temporary, index=False)
        temporary.replace(self.csv_path)
        logger.info(f"✓ Results aggregated in {self.csv_path}")

    @staticmethod
    def extract_metadata_from_config(config: Any) -> Dict[str, Any]:
        """Extract relevant metadata from a RunnerConfig object."""
        meta = {
            "model_id": config.model_id,
            "dataset_id": config.data_config.dataset_id,
            "run_id": config.run_id,
            "training_seed": config.training_config.seed,
            "dataset_revision": config.data_config.revision,
            "mmd_estimator": "unbiased_off_diagonal_v2",
            "signature_backend": "pysiglib",
            "signature_path_preparation": SIGNATURE_PATH_PREPARATION,
            "manifest": str(config.base_dir / "manifest.json"),
        }

        # Extract model specs if available
        if hasattr(config, "model_cfg"):
            cfg_dict = (
                config.model_cfg.model_dump()
                if hasattr(config.model_cfg, "model_dump")
                else vars(config.model_cfg)
            )
            for k, v in cfg_dict.items():
                if isinstance(v, (int, float, str, bool)):
                    meta[f"model_{k}"] = v

        return meta
