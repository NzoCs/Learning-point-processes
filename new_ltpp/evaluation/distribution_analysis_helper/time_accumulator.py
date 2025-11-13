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


class InterEventTimeAccumulator(BaseAccumulator):
    """Accumulates inter-event time statistics from batches."""

    def __init__(self, max_samples: Optional[int] = None):
        super().__init__(max_samples)
        self._gt_time_deltas: List[float] = []
        self._sim_time_deltas: List[float] = []

    def update(self, batch: Batch, simulation: Optional[SimulationResult] = None) -> None:
        """Accumulate inter-event times from batch.
        
        Args:
            batch: Ground truth batch containing time_delta_seqs
            simulation: Simulation result containing dtime_seqs
        """
        if not self.should_continue():
            return

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

            self._gt_time_deltas.extend(valid_deltas.tolist())
            self._sample_count += len(valid_deltas)

        # Process simulation if provided
        if simulation is not None:
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

                self._sim_time_deltas.extend(valid_sim_deltas.tolist())

    def compute(self) -> Dict[str, Any]:
        """Compute inter-event time statistics.
        
        Returns:
            Dictionary with 'gt_time_deltas' and 'sim_time_deltas' arrays
        """
        result = {
            'gt_time_deltas': np.array(self._gt_time_deltas),
            'sim_time_deltas': np.array(self._sim_time_deltas) if self._sim_time_deltas else np.array([]),
            'gt_count': len(self._gt_time_deltas),
            'sim_count': len(self._sim_time_deltas),
        }

        logger.info(f"InterEventTimeAccumulator: Collected {result['gt_count']} GT and {result['sim_count']} simulated time deltas")
        return result

    def reset(self) -> None:
        """Reset accumulated data."""
        super().reset()
        self._gt_time_deltas.clear()
        self._sim_time_deltas.clear()
