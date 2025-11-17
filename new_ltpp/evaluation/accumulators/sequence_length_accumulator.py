"""
Sequence Length Accumulator

Accumulates sequence length statistics from batches during prediction.
"""

from collections import Counter

import numpy as np
import torch

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .base_accumulator import BaseAccumulator
from .acc_types import SequenceLengthStatistics



class SequenceLengthAccumulator(BaseAccumulator):
    """Accumulates sequence length statistics."""

    def __init__(self, min_sim_events: int = 1):
        super().__init__(min_sim_events)
        # Use counters to store frequency of sequence lengths instead of raw lists
        self._gt_seq_lengths: Counter[int] = Counter()
        self._sim_seq_lengths: Counter[int] = Counter()

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate sequence lengths from batch.
        
        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)
        """

        # Validate simulation has sufficient events (vectorized torch operations)
        sim_types = simulation.type_seqs
        sim_seq_lengths = (sim_types > 0).sum(dim=1)
        sim_event_count = int(sim_seq_lengths.sum().item())

        if sim_event_count < self.min_sim_events:
            logger.error(
                "SequenceLengthAccumulator: Too few simulated events (%d < %d)",
                sim_event_count,
                self.min_sim_events,
            )
            raise ValueError(
                f"SequenceLengthAccumulator requires at least {self.min_sim_events} simulated events, got {sim_event_count}"
            )
        # Extract ground truth sequence lengths (vectorized)
        mask = batch.seq_non_pad_mask.bool()
        gt_seq_lengths = mask.sum(dim=1)

        # Extend counters using one CPU transfer per tensor
        gt_list = gt_seq_lengths.view(-1).cpu().tolist()
        for x in gt_list:
            self._gt_seq_lengths[int(x)] += 1
        self._sample_count += len(gt_list)

        # Process simulation sequence lengths (vectorized)
        sim_list = sim_seq_lengths.view(-1).cpu().tolist()
        for x in sim_list:
            self._sim_seq_lengths[int(x)] += 1

    def compute(self) -> SequenceLengthStatistics:
        """Compute sequence length statistics.
        
        Returns:
            Dictionary with sequence length arrays and statistics
        """
        # Validate sufficient ground truth data
        if sum(self._gt_seq_lengths.values()) == 0:
            logger.error("SequenceLengthAccumulator: No ground truth sequence lengths collected")
            raise ValueError("Cannot compute sequence length statistics: no ground truth data available")

        # Validate sufficient simulation data
        if sum(self._sim_seq_lengths.values()) == 0:
            logger.error("SequenceLengthAccumulator: No simulated sequence lengths collected")
            raise ValueError("Cannot compute sequence length statistics: no simulation data available")

        # Expand counters into arrays for statistics computation
        gt_lengths = np.array(list(self._gt_seq_lengths.keys()), dtype=int)
        gt_counts = np.array([self._gt_seq_lengths[int(k)] for k in gt_lengths], dtype=int)
        gt_array = np.repeat(gt_lengths, gt_counts)

        sim_lengths = np.array(list(self._sim_seq_lengths.keys()), dtype=int)
        sim_counts = np.array([self._sim_seq_lengths[int(k)] for k in sim_lengths], dtype=int)
        sim_array = np.repeat(sim_lengths, sim_counts)

        result = SequenceLengthStatistics(
            gt_array=gt_array,
            sim_array=sim_array,
            gt_mean=float(gt_array.mean()),
            gt_median=float(np.median(gt_array)),
            sim_mean=float(sim_array.mean()),
            sim_median=float(np.median(sim_array)),
            gt_count=len(gt_array),
            sim_count=len(sim_array),
        )

        logger.info(f"SequenceLengthAccumulator: Collected {result['gt_count']} GT and {result['sim_count']} simulated sequence lengths")
        return result

    def reset(self) -> None:
        """Reset accumulated data."""
        super().reset()
        self._gt_seq_lengths = Counter()
        self._sim_seq_lengths = Counter()
