"""
Mean Benchmark for Inter-Time Prediction

This benchmark always predicts the mean inter-time from the training dataset.
It computes RMSE and other time-based metrics using the metrics helper.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.utils import logger

from .base_bench import Benchmark, BenchmarkMode


class MeanInterTimeBenchmark(Benchmark):
    """
    Benchmark that predicts the mean inter-time for all events.
    """

    def __init__(
        self, data_config: DataConfig, dataset_name: str, save_dir: str = None
    ):
        """
        Initialize the mean inter-time benchmark.

        Args:
            data_config: Data configuration object
            dataset_name: Name of the dataset
            save_dir: Directory to save results
        """
        # This benchmark focuses on time prediction, so default to TIME_ONLY
        super().__init__(
            data_config, dataset_name, save_dir, benchmark_mode=BenchmarkMode.TIME_ONLY
        )
        self.mean_inter_time = None

    @property
    def benchmark_name(self) -> str:
        """Return the name of this benchmark."""
        return "mean_inter_time"

    def _prepare_benchmark(self) -> None:
        """
        Compute the mean inter-time from the training dataset.
        """
        train_loader = self.data_module.test_dataloader()

        logger.info("Computing mean inter-time from training data...")
        cumsum_inter_times = 0.0
        event_count = 0

        for batch in train_loader:
            # Extract inter-times from batch
            # batch structure: dict with keys: 'time_seqs', 'time_delta_seqs', 'type_seqs', 'batch_non_pad_mask', ...
            time_delta_seqs = batch["time_delta_seqs"]  # Inter-times
            batch_non_pad_mask = batch.get("batch_non_pad_mask", None)

            if batch_non_pad_mask is not None:
                # Only consider non-padded values
                mask = batch_non_pad_mask.bool()
                valid_inter_times = time_delta_seqs[mask]
            else:
                valid_inter_times = time_delta_seqs.flatten()

            cumsum_inter_times += valid_inter_times.sum().item()
            event_count += valid_inter_times.numel()

        self.mean_inter_time = (
            cumsum_inter_times / event_count if event_count > 0 else 0.0
        )
        logger.info(f"Computed mean inter-time: {self.mean_inter_time:.6f}")

    def _create_time_predictions(self, batch: Tuple) -> torch.Tensor:
        """
        Create time predictions using the mean inter-time.

        Args:
            batch: Input batch

        Returns:
            Tensor of predicted inter-times
        """
        time_delta_seqs = batch["time_delta_seqs"]

        # Create predictions with mean inter-time
        pred_inter_times = torch.full_like(time_delta_seqs, self.mean_inter_time)

        return pred_inter_times

    def _get_custom_results_info(self) -> Dict[str, Any]:
        """Add custom information specific to this benchmark."""
        return {
            "mean_inter_time_used": (
                float(self.mean_inter_time)
                if self.mean_inter_time is not None
                else None
            )
        }
