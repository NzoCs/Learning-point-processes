"""
Base Benchmark Class

This module provides a base class for implementing benchmarks for TPP models.
It defines the common interface and shared functionality that all benchmarks should implement.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.data.preprocess.data_loader import TPPDataModule
from new_ltpp.evaluation.benchmarks.bench_interfaces import BenchmarkInterface
from new_ltpp.evaluation.metrics_helper import MetricsHelper
from new_ltpp.globals import OUTPUT_DIR
from new_ltpp.utils import logger


class BaseBenchmark(ABC, BenchmarkInterface):
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
        self.metrics_helper = MetricsHelper(
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
        Save benchmark results to JSON file.

        Args:
            results: Results dictionary to save
        """
        # Create save directory
        dataset_dir = self.base_dir / self.data_config.dataset_id / "benchmarks"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        results_file = dataset_dir / f"{self.benchmark_name}_results.json"
        results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

        logger.info(f"Results saved to: {results_file}")

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

    def get_available_metrics(self) -> List[str]:
        """
        Get the list of available metrics for this benchmark.

        Returns:
            List of metric names
        """
        return self.metrics_helper.get_available_metrics()

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
            if "confusion" in metric_name.lower() or "matrix" in metric_name.lower():
                # For confusion matrices, sum them up instead of averaging
                try:
                    confusion_matrices = []
                    for m in all_metrics:
                        if metric_name in m:
                            value = m[metric_name]
                            if hasattr(value, "cpu"):
                                value = value.cpu().detach()
                            confusion_matrices.append(value)

                    if confusion_matrices:
                        # Sum all confusion matrices
                        if isinstance(confusion_matrices[0], torch.Tensor):
                            total_confusion = torch.stack(confusion_matrices).sum(dim=0)
                            aggregated[metric_name] = total_confusion.tolist()
                        else:
                            total_confusion = np.sum(confusion_matrices, axis=0)
                            aggregated[metric_name] = total_confusion.tolist()
                except Exception as e:
                    logger.warning(
                        f"Could not aggregate confusion matrix {metric_name}: {e}"
                    )
                continue

            # Extract values and convert tensors to scalars for regular metrics
            values = []
            for m in all_metrics:
                if metric_name in m:
                    value = m[metric_name]

                    # Handle different types of values
                    try:
                        # For torch tensors
                        if hasattr(value, "item"):
                            # Single element tensor
                            if value.numel() == 1:
                                value = value.item()
                            else:
                                # Multi-element tensor - convert to float and take mean
                                if value.dtype in [
                                    torch.long,
                                    torch.int,
                                    torch.int32,
                                    torch.int64,
                                ]:
                                    value = value.float()
                                value = float(value.mean().item())
                        elif hasattr(value, "cpu"):
                            # Tensor that needs to be moved to CPU first
                            cpu_value = value.cpu().detach()
                            if cpu_value.numel() == 1:
                                value = float(cpu_value.numpy())
                            else:
                                # Convert to float if integer type
                                if cpu_value.dtype in [
                                    torch.long,
                                    torch.int,
                                    torch.int32,
                                    torch.int64,
                                ]:
                                    cpu_value = cpu_value.float()
                                value = float(cpu_value.mean().numpy())
                        elif hasattr(value, "__len__") and len(value) > 1:
                            # Array-like object
                            value = float(np.mean(value))
                        else:
                            # Already a scalar
                            value = float(value)

                        # Check if value is not NaN
                        if not np.isnan(value):
                            values.append(value)
                    except Exception as e:
                        logger.warning(
                            f"Could not process metric {metric_name} with value {value}: {e}"
                        )
                        continue

            if values:
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[f"{metric_name}_std"] = float(np.std(values))
                aggregated[f"{metric_name}_min"] = float(np.min(values))
                aggregated[f"{metric_name}_max"] = float(np.max(values))
            else:
                aggregated[f"{metric_name}_mean"] = float("nan")
                aggregated[f"{metric_name}_std"] = float("nan")
                aggregated[f"{metric_name}_min"] = float("nan")
                aggregated[f"{metric_name}_max"] = float("nan")

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
        results = {
            "benchmark_name": self.benchmark_name,
            "num_event_types": self.num_event_types,
            "metrics": aggregated_metrics,
            "num_batches_evaluated": num_batches,
        }

        # Allow subclasses to add custom information
        custom_info = self._get_custom_results_info()
        if custom_info:
            results.update(custom_info)

        return results
