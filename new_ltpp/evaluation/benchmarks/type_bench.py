"""
Type Benchmark Base Class

This module provides the base class for type/mark prediction benchmarks.
"""

from abc import abstractmethod
from typing import Any, Dict

import torch

from new_ltpp.shared_types import Batch
from new_ltpp.utils import logger

from .base_bench import Benchmark


class TypeBenchmark(Benchmark):
    """
    Abstract base class for type/mark prediction benchmarks.

    This class handles benchmarks that focus on predicting event types/marks.
    """

    @abstractmethod
    def _create_type_predictions(self, batch: Batch) -> torch.Tensor:
        """
        Create type predictions for a given batch using the benchmark strategy.

        Args:
            batch: Input batch from the data loader

        Returns:
            Tensor of predicted types
        """
        pass

    def evaluate(self) -> Dict[str, Any]:
        """
        Run the type benchmark evaluation.

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
            # Only compute type metrics
            type_predictions = self._create_type_predictions(batch)
            metrics = self.metrics_helper.compute_prediction_type_metrics(
                batch, type_predictions
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
