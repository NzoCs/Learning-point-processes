"""
Last Mark Benchmark

This benchmark predicts the next event mark (type) using the previous event mark (lag-1).
"""

from typing import Any, Dict, Tuple

import torch
import yaml

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.utils import logger

from .base_bench import Benchmark, BenchmarkMode


class LastMarkBenchmark(Benchmark):
    """
    Benchmark that predicts the previous event mark as the next mark (lag-1).
    """

    def __init__(
        self, data_config: DataConfig, save_dir: str = None
    ):
        """
        Initialize the last mark benchmark.
        Args:
            data_config: Data configuration object
            save_dir: Directory to save results
        """
        super().__init__(
            data_config, save_dir, benchmark_mode=BenchmarkMode.TYPE_ONLY
        )

    def _create_type_predictions(self, batch: Tuple) -> torch.Tensor:
        """
        Create type predictions using the lag-1 mark strategy.

        Args:
            batch: Input batch

        Returns:
            Tensor of predicted types
        """
        type_seqs = batch["type_seqs"]
        batch_size, seq_len = type_seqs.shape

        # Create predictions for marks using lag-1 strategy
        pred_types = torch.zeros_like(type_seqs)
        safe_type_seqs = type_seqs.masked_fill(
            type_seqs == self.pad_token, 0
        )  # Avoid pad token issues

        # For positions 1 to seq_len-1, use the previous mark (lag-1)
        if seq_len > 1:
            pred_types[:, 1:] = safe_type_seqs[:, :-1]

        # For the first position (index 0), we cannot predict, so it will remain 0

        return pred_types

    @property
    def benchmark_name(self) -> str:
        return "lag1_mark_benchmark"

    def _prepare_benchmark(self) -> None:
        pass  # No special preparation needed for lag-1 strategy

    def _get_custom_results_info(self) -> Dict[str, Any]:
        """Add custom information specific to this benchmark."""
        return {"strategy": "lag1_mark_prediction"}
