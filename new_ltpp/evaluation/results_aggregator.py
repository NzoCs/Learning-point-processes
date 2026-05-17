from new_ltpp.evaluation.accumulators import FinalResult
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
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

        if self.csv_path.exists():
            try:
                existing_df = pd.read_csv(self.csv_path)
                # If experiment already exists, update it, otherwise append
                if experiment_id in existing_df["experiment_id"].values:
                    # Update: Merge with existing row to preserve columns from other phases
                    idx = existing_df[
                        existing_df["experiment_id"] == experiment_id
                    ].index[0]
                    for col in new_df.columns:
                        existing_df.at[idx, col] = new_df.at[0, col]
                    final_df = existing_df
                else:
                    final_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                logger.error(f"Error reading existing results CSV: {e}")
                final_df = new_df
        else:
            final_df = new_df

        # Save back to CSV
        final_df.to_csv(self.csv_path, index=False)
        logger.info(f"✓ Results aggregated in {self.csv_path}")

    @staticmethod
    def extract_metadata_from_config(config: Any) -> Dict[str, Any]:
        """Extract relevant metadata from a RunnerConfig object."""
        meta = {
            "model_id": config.model_id,
            "dataset_id": config.data_config.dataset_id,
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
