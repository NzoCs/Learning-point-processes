"""
Statistical Moments Accumulator

Accumulates statistical moments (mean, variance, skewness, kurtosis) using
online/incremental algorithms (Welford's method).
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .base_accumulator import BaseAccumulator


class MomentAccumulator(BaseAccumulator):
    """Accumulates statistical moments (mean, variance, skewness, kurtosis) online."""

    def __init__(self, max_samples: Optional[int] = None):
        super().__init__(max_samples)
        # For online computation of moments
        self._gt_n = 0
        self._gt_mean = 0.0
        self._gt_m2 = 0.0
        self._gt_m3 = 0.0
        self._gt_m4 = 0.0

        self._sim_n = 0
        self._sim_mean = 0.0
        self._sim_m2 = 0.0
        self._sim_m3 = 0.0
        self._sim_m4 = 0.0

    def update(self, batch: Batch, simulation: Optional[SimulationResult] = None) -> None:
        """Update moments using Welford's online algorithm.
        
        Args:
            batch: Ground truth batch
            simulation: Simulation result
        """
        if not self.should_continue():
            return

        # Extract ground truth time deltas
        time_deltas = batch.time_delta_seqs
        mask = batch.seq_non_pad_mask

        if torch.is_tensor(time_deltas):
            time_deltas = time_deltas.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # Flatten and filter valid values
        flat_deltas = time_deltas[mask.astype(bool)]
        flat_deltas = flat_deltas[flat_deltas > 0]

        # Update moments for ground truth
        for x in flat_deltas:
            if not self.should_continue():
                break
            self._update_moments_gt(x)
            self._sample_count += 1

        # Process simulation if provided
        if simulation is not None:
            sim_deltas = simulation.dtime_seqs
            if torch.is_tensor(sim_deltas):
                sim_deltas = sim_deltas.cpu().numpy()

            # Filter non-zero values (masked values are set to 0)
            flat_sim_deltas = sim_deltas.flatten()
            flat_sim_deltas = flat_sim_deltas[flat_sim_deltas > 0]

            for x in flat_sim_deltas:
                if not self.should_continue():
                    break
                self._update_moments_sim(x)

    def _update_moments_gt(self, x: float) -> None:
        """Update ground truth moments with new value using Welford's algorithm."""
        self._gt_n += 1
        n = self._gt_n

        delta = x - self._gt_mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * (n - 1)

        self._gt_mean += delta_n
        self._gt_m4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * self._gt_m2 - 4 * delta_n * self._gt_m3
        self._gt_m3 += term1 * delta_n * (n - 2) - 3 * delta_n * self._gt_m2
        self._gt_m2 += term1

    def _update_moments_sim(self, x: float) -> None:
        """Update simulation moments with new value using Welford's algorithm."""
        self._sim_n += 1
        n = self._sim_n

        delta = x - self._sim_mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * (n - 1)

        self._sim_mean += delta_n
        self._sim_m4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * self._sim_m2 - 4 * delta_n * self._sim_m3
        self._sim_m3 += term1 * delta_n * (n - 2) - 3 * delta_n * self._sim_m2
        self._sim_m2 += term1

    def compute(self) -> Dict[str, Any]:
        """Compute final statistical moments.
        
        Returns:
            Dictionary with mean, variance, std, skewness, kurtosis for GT and sim
        """
        # Ground truth statistics
        gt_variance = self._gt_m2 / self._gt_n if self._gt_n > 1 else 0.0
        gt_std = np.sqrt(gt_variance)
        gt_skewness = (np.sqrt(self._gt_n) * self._gt_m3) / (self._gt_m2 ** 1.5) if self._gt_m2 > 0 else 0.0
        gt_kurtosis = (self._gt_n * self._gt_m4) / (self._gt_m2 ** 2) - 3.0 if self._gt_m2 > 0 else 0.0

        # Simulation statistics
        sim_variance = self._sim_m2 / self._sim_n if self._sim_n > 1 else 0.0
        sim_std = np.sqrt(sim_variance)
        sim_skewness = (np.sqrt(self._sim_n) * self._sim_m3) / (self._sim_m2 ** 1.5) if self._sim_m2 > 0 else 0.0
        sim_kurtosis = (self._sim_n * self._sim_m4) / (self._sim_m2 ** 2) - 3.0 if self._sim_m2 > 0 else 0.0

        result = {
            'gt_mean': self._gt_mean,
            'gt_variance': gt_variance,
            'gt_std': gt_std,
            'gt_skewness': gt_skewness,
            'gt_kurtosis': gt_kurtosis,
            'gt_n': self._gt_n,
            'sim_mean': self._sim_mean,
            'sim_variance': sim_variance,
            'sim_std': sim_std,
            'sim_skewness': sim_skewness,
            'sim_kurtosis': sim_kurtosis,
            'sim_n': self._sim_n,
        }

        logger.info(f"MomentAccumulator: Computed moments from {self._gt_n} GT and {self._sim_n} simulated samples")
        return result

    def reset(self) -> None:
        """Reset accumulated moments."""
        super().reset()
        self._gt_n = 0
        self._gt_mean = 0.0
        self._gt_m2 = 0.0
        self._gt_m3 = 0.0
        self._gt_m4 = 0.0
        self._sim_n = 0
        self._sim_mean = 0.0
        self._sim_m2 = 0.0
        self._sim_m3 = 0.0
        self._sim_m4 = 0.0
