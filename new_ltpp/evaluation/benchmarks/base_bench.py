"""
Base Benchmark Class

This module provides a base class for implementing benchmarks for TPP models.
It defines the common interface and shared functionality that all benchmarks should implement.
"""

import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import yaml

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.data.preprocess.data_loader import TPPDataModule
from new_ltpp.evaluation.metrics_helper import EvaluationMode, MetricsHelper
from new_ltpp.utils import logger
from new_ltpp.evaluation.benchmarks.bench_interfaces import BenchmarkInterface


class BenchmarkMode(Enum):
    """Defines what type of predictions and metrics to compute."""

    TIME_ONLY = "time_only"
    TYPE_ONLY = "type_only"
    BOTH = "both"


class Benchmark(ABC, BenchmarkInterface):
    """
    Abstract base class for TPP benchmarks.

    This class provides the common structure and functionality that all benchmarks
    should inherit from. It handles data loading, metrics computation, result
    aggregation, and file saving.
    """

    def __init__(
        self,
        data_config: DataConfig,
        dataset_name: str,
        save_dir: str = None,
        benchmark_mode: str = BenchmarkMode.BOTH,
    ):
        """
        Initialize the base benchmark.

        Args:
            data_config: DataConfig object
            dataset_name: Name of the dataset
            save_dir: Directory to save results
            benchmark_mode: What to evaluate - "time_only", "type_only", or "both"
        """
        self.data_config = data_config
        self.save_dir = save_dir or "./benchmark_results"
        self.dataset_name = dataset_name
        self.benchmark_mode = benchmark_mode
        self.pad_token = data_config.tokenizer_specs.pad_token_id

        # Initialize data module with data_config
        self.data_module = TPPDataModule(data_config)
        self.data_module.setup("test")  # Setup test data

        # Initialize metrics helper
        self.metrics_helper = MetricsHelper(
            num_event_types=self.data_module.num_event_types,
            mode=EvaluationMode.PREDICTION,
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

    def _create_predictions(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create predictions for a given batch using the benchmark strategy.

        Args:
            batch: Input batch from the data loader

        Returns:
            Tuple of (predicted_inter_times, predicted_types)
        """
        pass

    def _create_time_predictions(self, batch: Tuple) -> torch.Tensor:
        """
        Create time predictions for a given batch using the benchmark strategy.

        Args:
            batch: Input batch from the data loader

        Returns:
            Tensor of predicted inter-times
        """
        # Default implementation uses the legacy method
        pred_times, _ = self._create_predictions(batch)
        return pred_times

    def _create_type_predictions(self, batch: Tuple) -> torch.Tensor:
        """
        Create type predictions for a given batch using the benchmark strategy.

        Args:
            batch: Input batch from the data loader

        Returns:
            Tensor of predicted types
        """
        # Default implementation uses the legacy method
        _, pred_types = self._create_predictions(batch)
        return pred_types

    def evaluate(self) -> Dict[str, Any]:
        """
        Run the benchmark evaluation.

        This method orchestrates the entire benchmark process:
        1. Prepare the benchmark (compute statistics, etc.)
        2. Evaluate on test data
        3. Aggregate metrics
        4. Save results

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting {self.benchmark_name} benchmark evaluation...")

        # Prepare benchmark-specific parameters
        self._prepare_benchmark()
        # Evaluate on test data
        test_loader = self.data_module.test_dataloader()
        all_metrics = []

        for batch_idx, batch in enumerate(test_loader):
            # Convert batch to values for compatibility
            batch_values = batch.values()

            # Compute metrics based on benchmark mode
            if self.benchmark_mode == BenchmarkMode.TIME_ONLY:
                # Only compute time metrics
                time_predictions = self._create_time_predictions(batch)
                metrics = self.metrics_helper.compute_all_time_metrics(
                    batch_values, time_predictions
                )

            elif self.benchmark_mode == BenchmarkMode.TYPE_ONLY:
                # Only compute type metrics
                type_predictions = self._create_type_predictions(batch)
                metrics = self.metrics_helper.compute_all_type_metrics(
                    batch_values, type_predictions
                )

            else:  # BenchmarkMode.BOTH
                # Compute all metrics (legacy behavior)
                pred_inter_times, pred_types = self._create_predictions(batch)
                predictions = (pred_inter_times, pred_types)
                metrics = self.metrics_helper.compute_all_metrics(
                    batch_values, predictions
                )

            all_metrics.append(metrics)

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")

        # Aggregate metrics across all batches
        aggregated_metrics = self._aggregate_metrics(all_metrics)

        # Prepare results
        results = self._prepare_results(aggregated_metrics, len(all_metrics))

        # Save results
        self._save_results(results)

        # Log summary
        self._log_summary(results)

        return results

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
            "dataset_name": self.dataset_name,
            "num_event_types": self.num_event_types,
            "metrics": aggregated_metrics,
            "num_batches_evaluated": num_batches,
        }

        # Allow subclasses to add custom information
        custom_info = self._get_custom_results_info()
        if custom_info:
            results.update(custom_info)

        return results

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
        dataset_dir = os.path.join(self.save_dir, self.dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Save results
        results_file = os.path.join(dataset_dir, f"{self.benchmark_name}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

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
