"""
Sequence Length Accumulator

Accumulates sequence length statistics from batches during prediction.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .base_accumulator import BaseAccumulator
from .types import SequenceLengthStatistics



class SequenceLengthAccumulator(BaseAccumulator):
    """Accumulates sequence length statistics."""

    def __init__(self, max_samples: Optional[int] = None, min_sim_events: int = 1):
        super().__init__(max_samples, min_sim_events)
        self._gt_seq_lengths: List[int] = []
        self._sim_seq_lengths: List[int] = []

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate sequence lengths from batch.
        
        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)
        """
        if not self.should_continue():
            return

        # Validate simulation has sufficient events
        sim_types = simulation.type_seqs
        if torch.is_tensor(sim_types):
            sim_event_count = int((sim_types > 0).sum().item())
        else:
            sim_event_count = int(np.count_nonzero(sim_types))
        
        if sim_event_count < self.min_sim_events:
            logger.error(
                "SequenceLengthAccumulator: Too few simulated events (%d < %d)",
                sim_event_count,
                self.min_sim_events,
            )
            raise ValueError(
                f"SequenceLengthAccumulator requires at least {self.min_sim_events} simulated events, got {sim_event_count}"
            )

        # Extract ground truth sequence lengths
        mask = batch.seq_non_pad_mask

        if torch.is_tensor(mask):
            mask = mask.cpu()

        # Calculate sequence lengths from mask
        batch_size = mask.shape[0]
        for i in range(batch_size):
            if not self.should_continue():
                break

            seq_length = int(mask[i].sum().item())
            self._gt_seq_lengths.append(seq_length)
            self._sample_count += 1

        # Process simulation (required)
        # Count non-zero events (masked values are set to 0)
        sim_types = simulation.type_seqs
        if torch.is_tensor(sim_types):
            sim_types = sim_types.cpu()

        for i in range(sim_types.shape[0]):
            if not self.should_continue():
                break

            sim_seq_length = int((sim_types[i] > 0).sum().item())
            self._sim_seq_lengths.append(sim_seq_length)

    def compute(self) -> SequenceLengthStatistics:
        """Compute sequence length statistics.
        
        Returns:
            Dictionary with sequence length arrays and statistics
        """
        # Validate sufficient ground truth data
        if len(self._gt_seq_lengths) == 0:
            logger.error("SequenceLengthAccumulator: No ground truth sequence lengths collected")
            raise ValueError("Cannot compute sequence length statistics: no ground truth data available")
        
        # Validate sufficient simulation data
        if len(self._sim_seq_lengths) == 0:
            logger.error("SequenceLengthAccumulator: No simulated sequence lengths collected")
            raise ValueError("Cannot compute sequence length statistics: no simulation data available")
        
        gt_array = np.array(self._gt_seq_lengths)
        sim_array = np.array(self._sim_seq_lengths)

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
        self._gt_seq_lengths.clear()
        self._sim_seq_lengths.clear()
