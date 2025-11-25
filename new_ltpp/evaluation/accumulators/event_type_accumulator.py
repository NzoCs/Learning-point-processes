"""
Event Type Accumulator

Accumulates event type distribution statistics from batches during prediction.
"""

from typing import List

import numpy as np

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .acc_types import EventTypeStatistics
from .base_accumulator import BaseAccumulator


class EventTypeAccumulator(BaseAccumulator):
    """Accumulates event type distribution statistics."""

    def __init__(self, num_event_types: int, min_sim_events: int = 1):
        super().__init__(min_sim_events)
        self.num_event_types = num_event_types
        self._gt_event_types: List[int] = []
        self._sim_event_types: List[int] = []

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate event types from batch.

        Args:
            batch: Ground truth batch containing type_seqs
            simulation: Simulation result containing type_seqs (required)
        """

        # Validate simulation has sufficient events (assumes simulation.type_seqs already masked)
        sim_types = simulation.type_seqs
        sim_event_count = sim_types.sum().item()
        if sim_event_count < self.min_sim_events:
            logger.error(
                "EventTypeAccumulator: Too few simulated events (%d < %d)",
                sim_event_count,
                self.min_sim_events,
            )
            raise ValueError(
                f"EventTypeAccumulator requires at least {self.min_sim_events} simulated events, got {sim_event_count}"
            )

        # Extract ground truth event types using torch (keeps tensors on device)
        type_seqs = batch.type_seqs
        mask = batch.seq_non_pad_mask

        gt_valid = type_seqs[mask.bool()]

        if gt_valid.numel() > 0:
            # Move to CPU only once for conversion to Python ints
            self._gt_event_types.extend(gt_valid.view(-1).cpu().tolist())
            self._sample_count += int(gt_valid.numel())

        # Process simulation vectorized (torch)
        valid_sim_types = sim_types[simulation.mask.bool()]
        if valid_sim_types.numel() > 0:
            self._sim_event_types.extend(valid_sim_types.view(-1).cpu().tolist())

    def compute(self) -> EventTypeStatistics:
        """Compute event type distribution statistics.

        Returns:
            Dictionary with event type arrays and distributions
        """
        # Validate sufficient ground truth data
        if len(self._gt_event_types) == 0:
            logger.error("EventTypeAccumulator: No ground truth event types collected")
            raise ValueError(
                "Cannot compute event type statistics: no ground truth data available"
            )

        # Validate sufficient simulation data
        if len(self._sim_event_types) == 0:
            logger.error("EventTypeAccumulator: No simulated event types collected")
            raise ValueError(
                "Cannot compute event type statistics: no simulation data available"
            )

        gt_array = np.array(self._gt_event_types, dtype=int)
        sim_array = np.array(self._sim_event_types, dtype=int)

        # Compute distributions
        gt_counts = np.bincount(gt_array, minlength=self.num_event_types + 1)
        gt_distribution = (gt_counts / gt_counts.sum()).astype(np.float64)

        sim_counts = np.bincount(sim_array, minlength=self.num_event_types + 1)
        sim_distribution = (sim_counts / sim_counts.sum()).astype(np.float64)

        result: EventTypeStatistics = EventTypeStatistics(
            gt_array=gt_array,
            sim_array=sim_array,
            gt_distribution=gt_distribution,
            sim_distribution=sim_distribution,
            gt_count=len(gt_array),
            sim_count=len(sim_array),
        )

        logger.info(
            f"EventTypeAccumulator: Collected {result['gt_count']} GT and {result['sim_count']} simulated event types"
        )
        return result

    def reset(self) -> None:
        """Reset accumulated data."""
        super().reset()
        self._gt_event_types.clear()
        self._sim_event_types.clear()
