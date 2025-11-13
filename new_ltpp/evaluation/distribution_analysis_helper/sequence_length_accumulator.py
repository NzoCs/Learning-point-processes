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


class SequenceLengthAccumulator(BaseAccumulator):
    """Accumulates sequence length statistics."""

    def __init__(self, max_samples: Optional[int] = None):
        super().__init__(max_samples)
        self._gt_seq_lengths: List[int] = []
        self._sim_seq_lengths: List[int] = []

    def update(self, batch: Batch, simulation: Optional[SimulationResult] = None) -> None:
        """Accumulate sequence lengths from batch.
        
        Args:
            batch: Ground truth batch
            simulation: Simulation result
        """
        if not self.should_continue():
            return

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

        # Process simulation if provided
        if simulation is not None:
            # Count non-zero events (masked values are set to 0)
            sim_types = simulation.type_seqs
            if torch.is_tensor(sim_types):
                sim_types = sim_types.cpu()

            for i in range(sim_types.shape[0]):
                if not self.should_continue():
                    break

                sim_seq_length = int((sim_types[i] > 0).sum().item())
                self._sim_seq_lengths.append(sim_seq_length)

    def compute(self) -> Dict[str, Any]:
        """Compute sequence length statistics.
        
        Returns:
            Dictionary with sequence length arrays and statistics
        """
        gt_array = np.array(self._gt_seq_lengths)
        sim_array = np.array(self._sim_seq_lengths) if self._sim_seq_lengths else np.array([])

        result = {
            'gt_seq_lengths': gt_array,
            'sim_seq_lengths': sim_array,
            'gt_mean': gt_array.mean() if len(gt_array) > 0 else 0.0,
            'gt_median': np.median(gt_array) if len(gt_array) > 0 else 0.0,
            'sim_mean': sim_array.mean() if len(sim_array) > 0 else 0.0,
            'sim_median': np.median(sim_array) if len(sim_array) > 0 else 0.0,
            'gt_count': len(gt_array),
            'sim_count': len(sim_array),
        }

        logger.info(f"SequenceLengthAccumulator: Collected {result['gt_count']} GT and {result['sim_count']} simulated sequence lengths")
        return result

    def reset(self) -> None:
        """Reset accumulated data."""
        super().reset()
        self._gt_seq_lengths.clear()
        self._sim_seq_lengths.clear()
