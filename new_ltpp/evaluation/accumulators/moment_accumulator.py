"""
Statistical Moments Accumulator

Accumulates statistical moments (mean, variance, skewness, kurtosis) using
online/incremental algorithms (Welford's method).
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from new_ltpp.evaluation.accumulators.types import MomentStatistics
from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .base_accumulator import BaseAccumulator


class MomentAccumulator(BaseAccumulator):
    """Accumulates statistical moments (mean, variance, skewness, kurtosis) online."""

    def __init__(self, max_samples: Optional[int] = None, min_sim_events: int = 1):
        super().__init__(max_samples, min_sim_events)
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
        self.name = "MomentAccumulator"

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Update moments using Welford's online algorithm.
        
        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)
        """
        if not self.should_continue():
            return

        # Validate simulation has sufficient events
        sim_deltas = simulation.dtime_seqs
        if torch.is_tensor(sim_deltas):
            sim_event_count = int((sim_deltas != 0).sum().item())
        else:
            sim_event_count = int(np.count_nonzero(sim_deltas))
        
        if sim_event_count < self.min_sim_events:
            logger.error(
                "MomentAccumulator: Too few simulated events (%d < %d)",
                sim_event_count,
                self.min_sim_events,
            )
            raise ValueError(
                f"MomentAccumulator requires at least {self.min_sim_events} simulated events, got {sim_event_count}"
            )

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

        # Process simulation (required)
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

    def compute(self) -> MomentStatistics:
        """Compute final statistical moments.
        
        Returns:
            Dictionary with mean, variance, std, skewness, kurtosis for GT and sim
        """
        # Validate sufficient ground truth data (need at least 2 samples for variance)
        if self._gt_n < 2:
            logger.error(f"MomentAccumulator: Insufficient ground truth samples ({self._gt_n} < 2)")
            raise ValueError(f"Cannot compute moment statistics: need at least 2 ground truth samples, got {self._gt_n}")
        
        # Validate sufficient simulation data
        if self._sim_n < 2:
            logger.error(f"MomentAccumulator: Insufficient simulated samples ({self._sim_n} < 2)")
            raise ValueError(f"Cannot compute moment statistics: need at least 2 simulated samples, got {self._sim_n}")
        
        # Ground truth statistics
        gt_variance = self._gt_m2 / self._gt_n
        gt_std = np.sqrt(gt_variance)
        gt_skewness = (np.sqrt(self._gt_n) * self._gt_m3) / (self._gt_m2 ** 1.5) if self._gt_m2 > 0 else 0.0
        gt_kurtosis = (self._gt_n * self._gt_m4) / (self._gt_m2 ** 2) - 3.0 if self._gt_m2 > 0 else 0.0

        # Simulation statistics
        sim_variance = self._sim_m2 / self._sim_n
        sim_std = np.sqrt(sim_variance)
        sim_skewness = (np.sqrt(self._sim_n) * self._sim_m3) / (self._sim_m2 ** 1.5) if self._sim_m2 > 0 else 0.0
        sim_kurtosis = (self._sim_n * self._sim_m4) / (self._sim_m2 ** 2) - 3.0 if self._sim_m2 > 0 else 0.0

        result = MomentStatistics(
            gt_mean=self._gt_mean,
            gt_variance=gt_variance,
            gt_std=gt_std,
            gt_skewness=gt_skewness,
            gt_kurtosis=gt_kurtosis,
            gt_n=self._gt_n,
            sim_mean=self._sim_mean,
            sim_variance=sim_variance,
            sim_std=sim_std,
            sim_skewness=sim_skewness,
            sim_kurtosis=sim_kurtosis,
            sim_n=self._sim_n,
        )

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
