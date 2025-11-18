"""
Mean Benchmark for Inter-Time Prediction

This benchmark always predicts the mean inter-time from the training dataset.
It computes RMSE and other time-based metrics using the metrics helper.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.shared_types import Batch
from new_ltpp.utils import logger

from .time_bench import TimeBenchmark


class MeanInterTimeBenchmark(TimeBenchmark):
    """
    Benchmark that predicts the mean inter-time for all events.
    """

    def __init__(self, data_config: DataConfig, base_dir: Union[str, Path]):
        """
        Initialize the mean inter-time benchmark.

        Args:
            data_config: Data configuration object
            base_dir: Directory to save results
        """
        super().__init__(data_config, base_dir)
        self.mean_inter_time = None

    @property
    def benchmark_name(self) -> str:
        """Return the name of this benchmark."""
        return "mean_inter_time"

    def _prepare_benchmark(self) -> None:
        """
        Compute the mean inter-time from the training dataset.
        """
        test_loader = self.data_module.test_dataloader()

        logger.info("Computing mean inter-time from test data...")
        cumsum_inter_times = 0.0
        event_count = 0

        for batch in test_loader:
            # Extract inter-event times from batch
            time_delta_seqs = batch.time_delta_seqs  # Inter-times
            batch_non_pad_mask = batch.seq_non_pad_mask

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

    def _create_dtime_predictions(self, batch: Batch) -> torch.Tensor:
        """
        Create time predictions using the mean inter-time.

        Args:
            batch: Input batch

        Returns:
            Tensor of predicted inter-event times
        """
        time_delta_seqs = batch.time_delta_seqs

        # Create predictions with mean inter-time
        if self.mean_inter_time is None:
            raise ValueError(
                "Mean inter-time has not been computed. Call _prepare_benchmark first."
            )

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
