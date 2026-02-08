"""
Event Type Accumulator

Accumulates event type distribution statistics from batches during prediction.
"""

from typing import List

import numpy as np

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .acc_types import EventTypeStatistics
from .base_accumulator import Accumulator


class EventTypeAccumulator(Accumulator):
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
            logger.warning(
                "EventTypeAccumulator: Too few simulated events (%d < %d), skipping batch",
                sim_event_count,
                self.min_sim_events,
            )
            return

        # Extract ground truth event types using torch (keeps tensors on device)
        type_seqs = batch.type_seqs
        mask = batch.valid_event_mask

        gt_valid = type_seqs[mask.bool()]

        if gt_valid.numel() > 0:
            # Move to CPU only once for conversion to Python ints
            self._gt_event_types.extend(gt_valid.view(-1).cpu().tolist())
            self._sample_count += int(gt_valid.numel())

        # Process simulation vectorized (torch)
        valid_sim_types = sim_types[simulation.valid_event_mask.bool()]
        if valid_sim_types.numel() > 0:
            self._sim_event_types.extend(valid_sim_types.view(-1).cpu().tolist())

    def compute(self) -> EventTypeStatistics:  # type: ignore[override]
        """Compute event type distribution statistics.

        Returns:
            Dictionary with event type arrays and distributions
        """
        # Handle case with no ground truth data
        if len(self._gt_event_types) == 0:
            logger.warning(
                "EventTypeAccumulator: No ground truth event types collected, returning empty statistics"
            )
            gt_array = np.array([], dtype=int)
        else:
            gt_array = np.array(self._gt_event_types, dtype=int)

        # Handle case with no simulation data
        if len(self._sim_event_types) == 0:
            logger.warning(
                "EventTypeAccumulator: No simulated event types collected, returning empty statistics"
            )
            sim_array = np.array([], dtype=int)
        else:
            sim_array = np.array(self._sim_event_types, dtype=int)

        # Compute distributions
        if len(gt_array) > 0:
            gt_counts = np.bincount(gt_array, minlength=self.num_event_types + 1)
            gt_distribution = (gt_counts / gt_counts.sum()).astype(np.float64)
        else:
            gt_counts = np.zeros(self.num_event_types + 1, dtype=int)
            gt_distribution = np.zeros(self.num_event_types + 1, dtype=np.float64)

        if len(sim_array) > 0:
            sim_counts = np.bincount(sim_array, minlength=self.num_event_types + 1)
            sim_distribution = (sim_counts / sim_counts.sum()).astype(np.float64)
        else:
            sim_counts = np.zeros(self.num_event_types + 1, dtype=int)
            sim_distribution = np.zeros(self.num_event_types + 1, dtype=np.float64)

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
