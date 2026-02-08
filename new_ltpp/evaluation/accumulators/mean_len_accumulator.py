"""
Sequence Length Accumulator

Accumulates sequence length statistics from batches during prediction.
"""

import numpy as np

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .acc_types import SequenceLengthStatistics
from .base_accumulator import Accumulator


class SequenceLengthAccumulator(Accumulator):
    """Accumulates sequence length statistics."""

    def __init__(self, min_sim_events: int = 1):
        super().__init__(min_sim_events)
        # Use counters to store frequency of sequence lengths instead of raw lists
        self._gt_mean: list[float] = []
        self._sim_mean: list[float] = []

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate sequence lengths from batch.

        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)
        """

        # Validate simulation has sufficient events (vectorized torch operations)
        sim_mask = simulation.valid_event_mask.bool()
        sim_seq_lengths = sim_mask.sum(dim=1)
        sim_event_count = int(sim_seq_lengths.sum().item())

        if sim_event_count < self.min_sim_events:
            logger.warning(
                "SequenceLengthAccumulator: Too few simulated events (%d < %d), skipping batch",
                sim_event_count,
                self.min_sim_events,
            )
            return

        # Extract simulated sequence lengths (vectorized)

        time_seqs = simulation.time_seqs

        time_seqs[~sim_mask] = float("inf")
        time_starts = time_seqs.min(dim=1).values

        time_seqs[~sim_mask] = 0.0
        time_ends = time_seqs.max(dim=1).values

        sim_time_windows = time_ends - time_starts
        sim_event_count_normalized = sim_event_count / sim_time_windows

        self._sim_mean.extend(sim_event_count_normalized.view(-1).cpu().tolist())

        # Extract ground truth sequence lengths (vectorized)
        mask = batch.valid_event_mask.bool()
        gt_seq_lengths = mask.sum(dim=1)

        gt_time_seqs = batch.time_seqs

        gt_time_seqs[~mask] = float("inf")
        gt_time_starts = gt_time_seqs.min(dim=1).values

        gt_time_seqs[~mask] = 0.0
        gt_time_ends = gt_time_seqs.max(dim=1).values

        gt_time_windows = gt_time_ends - gt_time_starts
        gt_event_count_normalized = gt_seq_lengths.sum().item() / gt_time_windows

        self._gt_mean.extend(gt_event_count_normalized.view(-1).cpu().tolist())

    def compute(self) -> SequenceLengthStatistics:  # type: ignore[override]
        """Compute sequence length statistics.

        Returns:
            Dictionary with sequence length arrays and statistics
        """

        # Handle cases with no data gracefully
        if len(self._gt_mean) == 0:
            logger.warning(
                "SequenceLengthAccumulator: No ground truth sequence lengths collected, returning empty statistics"
            )
            gt_lengths = np.array([], dtype=float)
        else:
            gt_lengths = np.array(self._gt_mean, dtype=float)

        if len(self._sim_mean) == 0:
            logger.warning(
                "SequenceLengthAccumulator: No simulated sequence lengths collected, returning empty statistics"
            )
            sim_lengths = np.array([], dtype=float)
        else:
            sim_lengths = np.array(self._sim_mean, dtype=float)

        result = SequenceLengthStatistics(
            gt_array=gt_lengths,
            sim_array=sim_lengths,
            gt_mean=float(gt_lengths.mean()) if len(gt_lengths) > 0 else 0.0,
            gt_median=float(np.median(gt_lengths)) if len(gt_lengths) > 0 else 0.0,
            sim_mean=float(sim_lengths.mean()) if len(sim_lengths) > 0 else 0.0,
            sim_median=float(np.median(sim_lengths)) if len(sim_lengths) > 0 else 0.0,
            gt_count=len(gt_lengths),
            sim_count=len(sim_lengths),
        )

        logger.info(
            f"SequenceLengthAccumulator: Collected {result['gt_count']} GT and {result['sim_count']} simulated sequence lengths"
        )
        return result

    def reset(self) -> None:
        """Reset accumulated data."""
        super().reset()
        self._gt_mean = []
        self._sim_mean = []
