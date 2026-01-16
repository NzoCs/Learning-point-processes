"""
Last Mark Benchmark

This benchmark predicts the next event mark (type) using the previous event mark (lag-1).
"""

from typing import Any, Dict

import torch

from new_ltpp.shared_types import Batch

from .type_bench import TypeBenchmark


class LastMarkBenchmark(TypeBenchmark):
    """
    Benchmark that predicts the previous event mark as the next mark (lag-1).
    """

    def _create_type_predictions(self, batch: Batch) -> torch.Tensor:
        """
        Create type predictions using the lag-1 mark strategy.

        Args:
            batch: Input batch

        Returns:
            Tensor of predicted types
        """
        type_seqs = batch.type_seqs
        batch_size, seq_len = type_seqs.shape

        # Create predictions for marks using lag-1 strategy
        pred_types = torch.zeros_like(type_seqs)
        mask = batch.valid_event_mask
        safe_type_seqs = type_seqs.masked_fill(
            ~mask, 0
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
