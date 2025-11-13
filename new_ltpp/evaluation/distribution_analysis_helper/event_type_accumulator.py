"""
Event Type Accumulator

Accumulates event type distribution statistics from batches during prediction.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .base_accumulator import BaseAccumulator


class EventTypeAccumulator(BaseAccumulator):
    """Accumulates event type distribution statistics."""

    def __init__(self, num_event_types: int, max_samples: Optional[int] = None):
        super().__init__(max_samples)
        self.num_event_types = num_event_types
        self._gt_event_types: List[int] = []
        self._sim_event_types: List[int] = []

    def update(self, batch: Batch, simulation: Optional[SimulationResult] = None) -> None:
        """Accumulate event types from batch.
        
        Args:
            batch: Ground truth batch containing type_seqs
            simulation: Simulation result containing type_seqs
        """
        if not self.should_continue():
            return

        # Extract ground truth event types
        type_seqs = batch.type_seqs
        mask = batch.seq_non_pad_mask

        if torch.is_tensor(type_seqs):
            type_seqs = type_seqs.cpu()
        if torch.is_tensor(mask):
            mask = mask.cpu()

        # Process each sequence in batch
        batch_size = type_seqs.shape[0]
        for i in range(batch_size):
            if not self.should_continue():
                break

            valid_mask = mask[i].bool() if mask is not None else torch.ones_like(type_seqs[i], dtype=torch.bool)
            valid_types = type_seqs[i][valid_mask].numpy()

            # Filter padding tokens (assuming -1 or 0 is padding)
            valid_types = valid_types[valid_types > 0]

            self._gt_event_types.extend(valid_types.tolist())
            self._sample_count += len(valid_types)

        # Process simulation if provided
        if simulation is not None:
            sim_types = simulation.type_seqs
            if torch.is_tensor(sim_types):
                sim_types = sim_types.cpu()

            for i in range(sim_types.shape[0]):
                if not self.should_continue():
                    break

                # Filter non-zero values (masked values are set to 0)
                valid_sim_types = sim_types[i].numpy()
                valid_sim_types = valid_sim_types[valid_sim_types > 0]

                self._sim_event_types.extend(valid_sim_types.tolist())

    def compute(self) -> Dict[str, Any]:
        """Compute event type distribution statistics.
        
        Returns:
            Dictionary with event type arrays and distributions
        """
        gt_array = np.array(self._gt_event_types)
        sim_array = np.array(self._sim_event_types) if self._sim_event_types else np.array([])

        # Compute distributions
        gt_counts = np.bincount(gt_array, minlength=self.num_event_types + 1)
        gt_distribution = gt_counts / gt_counts.sum() if gt_counts.sum() > 0 else gt_counts

        sim_counts = np.bincount(sim_array, minlength=self.num_event_types + 1) if len(sim_array) > 0 else np.zeros(self.num_event_types + 1)
        sim_distribution = sim_counts / sim_counts.sum() if sim_counts.sum() > 0 else sim_counts

        result = {
            'gt_event_types': gt_array,
            'sim_event_types': sim_array,
            'gt_distribution': gt_distribution,
            'sim_distribution': sim_distribution,
            'gt_count': len(gt_array),
            'sim_count': len(sim_array),
        }

        logger.info(f"EventTypeAccumulator: Collected {result['gt_count']} GT and {result['sim_count']} simulated event types")
        return result

    def reset(self) -> None:
        """Reset accumulated data."""
        super().reset()
        self._gt_event_types.clear()
        self._sim_event_types.clear()
