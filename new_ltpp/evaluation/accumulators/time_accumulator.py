"""
Inter-Event Time Accumulator

Accumulates inter-event time statistics from batches during prediction.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .base_accumulator import BaseAccumulator
from .types import TimeStatistics

class InterEventTimeAccumulator(BaseAccumulator):
    """Accumulates inter-event time statistics from batches using histogram bins."""

    def __init__(self, max_time: float, max_samples: Optional[int] = None, min_sim_events: int = 1, num_bins: int = 100):
        super().__init__(max_samples, min_sim_events)
        self.num_bins = num_bins
        self.max_time = max_time
        self.bin_edges = np.linspace(0, max_time, num_bins + 1)
        
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

        if not self.should_continue():
            return

        # Validate simulation has sufficient events
        sim_deltas = simulation.get_masked_dtime_values()
        sim_event_count = int((sim_deltas != 0).sum().item())
        
        if sim_event_count < self.min_sim_events:
            raise ValueError(
                f"InterEventTimeAccumulator requires at least {self.min_sim_events} simulated events, got {sim_event_count}"
            )

        # Extract ground truth time deltas
        time_deltas = batch.time_delta_seqs
        mask = batch.seq_non_pad_mask

        if torch.is_tensor(time_deltas):
            time_deltas = time_deltas.cpu()
        if torch.is_tensor(mask):
            mask = mask.cpu()

        # Process each sequence in batch
        batch_size = time_deltas.shape[0]
        for i in range(batch_size):
            if not self.should_continue():
                break

            valid_mask = mask[i].bool() if mask is not None else torch.ones_like(time_deltas[i], dtype=torch.bool)
            valid_deltas = time_deltas[i][valid_mask].numpy()

            # Filter out invalid values
            valid_deltas = valid_deltas[valid_deltas > 0]
            
            # Assign to histogram bins
            bin_indices = np.digitize(valid_deltas, self.bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
            
            for bin_idx in bin_indices:
                self._gt_hist[bin_idx] += 1
            
            self._gt_total += len(valid_deltas)
            self._sample_count += len(valid_deltas)

        # Process simulation (required)
        sim_deltas = simulation.dtime_seqs
        if torch.is_tensor(sim_deltas):
            sim_deltas = sim_deltas.cpu()

        # Use ground truth mask for simulation (sequences have same structure)
        for i in range(sim_deltas.shape[0]):
            if not self.should_continue():
                break

            # Filter non-zero values (masked values are set to 0)
            valid_sim_deltas = sim_deltas[i].numpy()
            valid_sim_deltas = valid_sim_deltas[valid_sim_deltas > 0]
            
            # Assign to histogram bins
            bin_indices = np.digitize(valid_sim_deltas, self.bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
            
            for bin_idx in bin_indices:
                self._sim_hist[bin_idx] += 1
            
            self._sim_total += len(valid_sim_deltas)

    def compute(self) -> TimeStatistics:
        """Compute inter-event time statistics.
        
        Returns:
            Dictionary with histogram bins and counts
        """
        # Validate sufficient ground truth data
        if self._gt_total == 0:
            logger.error("InterEventTimeAccumulator: No ground truth time deltas collected")
            raise ValueError("Cannot compute time statistics: no ground truth data available")
        
        # Validate sufficient simulation data
        if self._sim_total == 0:
            logger.error("InterEventTimeAccumulator: No simulated time deltas collected")
            raise ValueError("Cannot compute time statistics: no simulation data available")
        
        result = TimeStatistics(
            gt_time_deltas=self._gt_hist.astype(np.float64),
            sim_time_deltas=self._sim_hist.astype(np.float64),
            bin_edges=self.bin_edges,
            gt_count=self._gt_total,
            sim_count=self._sim_total,
        )

        logger.info(f"InterEventTimeAccumulator: Collected {result['gt_count']} GT and {result['sim_count']} simulated time deltas in {self.num_bins} bins")
        return result

    def reset(self) -> None:
        """Reset accumulated data."""
        super().reset()
        self._gt_hist.fill(0)
        self._sim_hist.fill(0)
        self._gt_total = 0
        self._sim_total = 0