"""
Inter-Event Time Accumulator

Accumulates inter-event time statistics from batches during prediction.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .acc_types import TimeStatistics
from .base_accumulator import BaseAccumulator


class InterEventTimeAccumulator(BaseAccumulator):
    """Accumulates inter-event time statistics from batches using histogram bins."""

    def __init__(
        self,
        dtime_min: float,
        dtime_max: float,
        min_sim_events: int = 1,
        num_bins: int = 100,
    ):
        super().__init__(min_sim_events)
        self.num_bins = num_bins
        self.min_dtime = dtime_min
        self.max_dtime = dtime_max
        self.bin_edges = np.linspace(dtime_min, dtime_max, num_bins + 1)

        # Histogram counters
        self._gt_hist = np.zeros(num_bins, dtype=np.int64)
        self._sim_hist = np.zeros(num_bins, dtype=np.int64)

        # Keep track of total counts
        self._gt_total = 0
        self._sim_total = 0

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate inter-event times from batch.

        Args:
            batch: Ground truth batch containing time_delta_seqs
            simulation: Simulation result containing dtime_seqs (required)
        """

        # Apply mask and flatten values for vectorized histogram computation.
        # Ground truth: select valid (non-padded) deltas > 0
        time_deltas = batch.time_delta_seqs
        mask = batch.seq_non_pad_mask.bool()
        valid_gt = time_deltas[mask]
        valid_gt = valid_gt[valid_gt > 0]

        # Simulation: use simulation mask if provided, otherwise assume masked values are <= 0
        sim_deltas = simulation.dtime_seqs
        # simulation.mask convention: True indicates masked positions (as used elsewhere)
        if hasattr(simulation, "mask") and simulation.mask is not None:
            sim_mask = simulation.mask.bool()
            valid_sim = sim_deltas[~sim_mask]
        else:
            valid_sim = sim_deltas.view(-1)
        valid_sim = valid_sim[valid_sim > 0]

        # Quick validation
        sim_event_count = int(valid_sim.numel())
        if sim_event_count < self.min_sim_events:
            logger.warning(
                "InterEventTimeAccumulator: Too few simulated events (%d < %d), skipping batch",
                sim_event_count,
                self.min_sim_events,
            )
            return

        # Prepare bin edges as torch tensor on same device
        device = valid_gt.device if valid_gt.numel() > 0 else valid_sim.device
        bin_edges_t = torch.tensor(self.bin_edges, device=device, dtype=valid_gt.dtype)

        # Compute bin indices and counts for ground truth using torch
        if valid_gt.numel() > 0:
            gt_bins = torch.bucketize(valid_gt, bin_edges_t) - 1
            gt_bins = gt_bins.clamp(0, self.num_bins - 1).to(torch.long)
            gt_counts = torch.bincount(gt_bins, minlength=self.num_bins)
            self._gt_hist += gt_counts.cpu().numpy()
            self._gt_total += int(valid_gt.numel())
            self._sample_count += int(valid_gt.numel())

        # Compute bin indices and counts for simulation using torch
        if valid_sim.numel() > 0:
            sim_bins = torch.bucketize(valid_sim, bin_edges_t) - 1
            sim_bins = sim_bins.clamp(0, self.num_bins - 1).to(torch.long)
            sim_counts = torch.bincount(sim_bins, minlength=self.num_bins)
            self._sim_hist += sim_counts.cpu().numpy()
            self._sim_total += int(valid_sim.numel())

    def compute(self) -> TimeStatistics:
        """Compute inter-event time statistics.

        Returns:
            Dictionary with histogram bins and counts
        """
        # Handle cases with no data gracefully
        if self._gt_total == 0:
            logger.warning(
                "InterEventTimeAccumulator: No ground truth time deltas collected, returning empty statistics"
            )
        
        if self._sim_total == 0:
            logger.warning(
                "InterEventTimeAccumulator: No simulated time deltas collected, returning empty statistics"
            )

        result = TimeStatistics(
            gt_time_deltas=self._gt_hist.astype(np.float64),
            sim_time_deltas=self._sim_hist.astype(np.float64),
            bin_edges=self.bin_edges,
            gt_count=self._gt_total,
            sim_count=self._sim_total,
        )

        logger.info(
            f"InterEventTimeAccumulator: Collected {result['gt_count']} GT and {result['sim_count']} simulated time deltas in {self.num_bins} bins"
        )
        return result

    def reset(self) -> None:
        """Reset accumulated data."""
        super().reset()
        self._gt_hist.fill(0)
        self._sim_hist.fill(0)
        self._gt_total = 0
        self._sim_total = 0
