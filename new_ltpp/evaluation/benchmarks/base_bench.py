"""
Base Benchmark Class

This module provides a base class for implementing benchmarks for TPP models.
It defines the common interface and shared functionality that all benchmarks should implement.
"""

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union, Protocol, runtime_checkable

import numpy as np
import torch

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.data.preprocess.data_loader import TPPDataModule
from new_ltpp.evaluation.metrics_helper import MetricsManager
from new_ltpp.globals import OUTPUT_DIR
from new_ltpp.utils import logger


class Benchmark(ABC):
    """
    Abstract base class for all TPP benchmarks.

    This class provides the common structure and functionality that all benchmarks
    should inherit from. It handles data loading, metrics computation, result
    aggregation, and file saving.
    """

    def __init__(
        self,
        data_config: DataConfig,
        base_dir: Union[str, Path] = OUTPUT_DIR,
    ):
        """
        Initialize the base benchmark.

        Args:
            data_config: DataConfig object
            save_dir: Directory to save results
        """
        self.data_config = data_config
        self.base_dir = Path(base_dir)
        self.pad_token = data_config.tokenizer_specs.pad_token_id

        # Initialize data module with data_config
        self.data_module = TPPDataModule(data_config)
        self.data_module.setup("test")  # Setup test data

        # Initialize metrics helper
        self.metrics_helper = MetricsManager(
            num_event_types=self.data_module.num_event_types,
        )

        self.num_event_types = self.data_module.num_event_types

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """
        Return the name of this benchmark.

        Returns:
            String identifier for this benchmark
        """
        pass

    @abstractmethod
    def _prepare_benchmark(self) -> None:
        """
        Prepare the benchmark by computing any necessary statistics or parameters
        from the training data. This method should be implemented by subclasses
        to perform benchmark-specific preparation.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Run the benchmark evaluation.

        This method orchestrates the entire benchmark process and must be
        implemented by subclasses to define the evaluation logic.

        Returns:
            Dictionary containing evaluation results
        """
        pass

    def _get_custom_results_info(self) -> Dict[str, Any]:
        """
        Get custom information to add to results.
        Subclasses can override this to add benchmark-specific information.

        Returns:
            Dictionary with custom information
        """
        return {}

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save benchmark results to a global CSV file.

        Args:
            results: Results dictionary to save
        """
        # Global CSV file in base_dir (usually artifacts directory)
        csv_file = self.base_dir / "benchmarks_results.csv"
        
        flat_results = {}
        flat_results["dataset_id"] = self.data_config.dataset_id
        flat_results["benchmark_name"] = results.get("benchmark_name", self.benchmark_name)
        flat_results["num_event_types"] = results.get("num_event_types", self.num_event_types)
        flat_results["num_batches_evaluated"] = results.get("num_batches_evaluated", 0)
        
        # Extract metrics
        metrics = results.get("metrics", {})
        for k, v in metrics.items():
            flat_results[k] = v
            
        # Extract any custom info
        for k, v in results.items():
            if k not in ["benchmark_name", "num_event_types", "metrics", "num_batches_evaluated"]:
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat_results[f"{k}_{sub_k}"] = sub_v
                else:
                    flat_results[k] = v

        # Read existing data if file exists
        existing_data = []
        fieldnames = list(flat_results.keys())
        
        if csv_file.exists():
            try:
                with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        for field in reader.fieldnames:
                            if field not in fieldnames:
                                fieldnames.append(field)
                    for row in reader:
                        existing_data.append(row)
            except Exception as e:
                logger.error(f"Error reading existing CSV {csv_file}: {e}")
                
        # Append new row
        existing_data.append(flat_results)
        
        # Rewrite the entire CSV with updated fieldnames
        ordered_fieldnames = []
        for primary_field in ["dataset_id", "benchmark_name"]:
            if primary_field in fieldnames:
                ordered_fieldnames.append(primary_field)
                fieldnames.remove(primary_field)
        ordered_fieldnames.extend(sorted(fieldnames))
        
        # Ensure directory exists just in case
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)

        logger.info(f"Results appended to: {csv_file}")

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """
        Log a summary of the benchmark results.

        Args:
            results: Results dictionary
        """
        logger.info(f"{self.benchmark_name} benchmark completed successfully!")

        metrics = results.get("metrics", {})

        # Log time-based metrics if available
        if "time_rmse_mean" in metrics:
            logger.info(f"Time RMSE: {metrics['time_rmse_mean']:.6f}")
        if "time_mae_mean" in metrics:
            logger.info(f"Time MAE: {metrics['time_mae_mean']:.6f}")

        # Log type-based metrics if available
        if "type_accuracy_mean" in metrics:
            logger.info(f"Type Accuracy: {metrics['type_accuracy_mean']:.6f}")
        if "macro_f1score_mean" in metrics:
            logger.info(f"Macro F1 Score: {metrics['macro_f1score_mean']:.6f}")

    def _aggregate_metrics(
        self, all_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across all batches.

        Args:
            all_metrics: List of metric dictionaries from each batch

        Returns:
            Aggregated metrics with mean, std, min, max for each metric
        """
        if not all_metrics:
            return {}
        # Get all metric names
        metric_names = all_metrics[0].keys()
        aggregated = {}
        for metric_name in metric_names:
            # Skip aggregation for non-scalar metrics like confusion matrices
            # Skip any confusion-matrix / matrix-like metrics entirely.
            if "confusion" in metric_name.lower() or "matrix" in metric_name.lower():
                logger.debug(f"Skipping confusion/matrix metric '{metric_name}'")
                continue

            # Extract values and convert tensors to scalars for regular metrics
            values = []
            for m in all_metrics:
                if metric_name in m:
                    value = m[metric_name]

                    # Handle different types of values
                    try:
                        # Torch tensor -> numpy
                        if isinstance(value, torch.Tensor):
                            arr = value.detach().cpu().numpy()
                            value = (
                                float(np.mean(arr)) if arr.size > 0 else float("nan")
                            )
                        # Numpy array or sequence -> numpy
                        elif isinstance(value, (np.ndarray, list, tuple)):
                            arr = np.array(value)
                            value = (
                                float(np.mean(arr)) if arr.size > 0 else float("nan")
                            )
                        else:
                            # Scalar-like (int/float)
                            value = float(value)

                        if not np.isnan(value):
                            values.append(value)
                    except Exception as e:
                        logger.warning(
                            f"Could not process metric {metric_name} with value {value}: {e}"
                        )
                        continue

            # Only keep the mean when aggregating to simplify outputs.
            if values:
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            else:
                aggregated[f"{metric_name}_mean"] = float("nan")

        return aggregated

    def _prepare_results(
        self, aggregated_metrics: Dict[str, float], num_batches: int
    ) -> Dict[str, Any]:
        """
        Prepare the results dictionary with benchmark information.

        Args:
            aggregated_metrics: Aggregated metrics
            num_batches: Number of batches processed

        Returns:
            Results dictionary
        """
        # Keep only the mean variants of each metric when saving benchmark results.
        mean_metrics = {
            k: v for k, v in aggregated_metrics.items() if k.endswith("_mean")
        }

        results = {
            "benchmark_name": self.benchmark_name,
            "num_event_types": self.num_event_types,
            "metrics": mean_metrics,
            "num_batches_evaluated": num_batches,
        }

        # Allow subclasses to add custom information
        custom_info = self._get_custom_results_info()
        if custom_info:
            results.update(custom_info)

        return results


@runtime_checkable
class IBenchmark(Protocol):
    """Protocol for IDE type checking + isinstance() support."""

    @property
    def benchmark_name(self) -> str:
        """Return the name of this benchmark."""
        ...

    def evaluate(self) -> Dict[str, Any]:
        """Run the benchmark evaluation."""
        ...
