"""
Metrics Calculator for Temporal Point Process Analysis.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import numpy.typing as npt

from .acc_types import MetricsData


class MetricsCalculatorImpl:
    """Calculates summary metrics from temporal point process data (SRP)."""

    def calculate_metrics(
        self, 
        data: MetricsData
    ) -> Dict[str, float]:
        """Calculate comprehensive summary metrics.
        
        Args:
            data: Dictionary containing arrays for:
                - label_time_deltas: Ground truth time deltas
                - simulated_time_deltas: Simulated time deltas
                - label_sequence_lengths: Ground truth sequence lengths
                - simulated_sequence_lengths: Simulated sequence lengths
                
        Returns:
            Dictionary of computed metrics
        """
        metrics: Dict[str, float] = {}

        # Inter-event time metrics
        label_deltas = data["label_time_deltas"]
        simulated_deltas = data["simulated_time_deltas"]

        if len(label_deltas) > 0:
            metrics.update(
                {
                    "ground_truth_mean_inter_event_time": float(
                        np.mean(label_deltas)
                    ),
                    "ground_truth_median_inter_event_time": float(
                        np.median(label_deltas)
                    ),
                    "ground_truth_std_inter_event_time": float(
                        np.std(label_deltas)
                    ),
                }
            )

        if len(simulated_deltas) > 0:
            metrics.update(
                {
                    "simulation_mean_inter_event_time": float(
                        np.mean(simulated_deltas)
                    ),
                    "simulation_median_inter_event_time": float(
                        np.median(simulated_deltas)
                    ),
                    "simulation_std_inter_event_time": float(
                        np.std(simulated_deltas)
                    ),
                }
            )

        # Sequence length metrics
        label_lengths = data["label_sequence_lengths"]
        simulated_lengths = data["simulated_sequence_lengths"]

        if len(label_lengths) > 0:
            metrics.update(
                {
                    "ground_truth_mean_sequence_length": float(
                        np.mean(label_lengths)
                    ),
                    "ground_truth_median_sequence_length": float(
                        np.median(label_lengths)
                    ),
                }
            )

        if len(simulated_lengths) > 0:
            metrics.update(
                {
                    "simulation_mean_sequence_length": float(
                        np.mean(simulated_lengths)
                    ),
                    "simulation_median_sequence_length": float(
                        np.median(simulated_lengths)
                    ),
                }
            )

        # Calculate relative differences
        if len(label_deltas) > 0 and len(simulated_deltas) > 0:
            gt_mean = np.mean(label_deltas)
            sim_mean = np.mean(simulated_deltas)
            metrics["relative_mean_time_difference"] = float(
                (sim_mean - gt_mean) / gt_mean
            )

        return metrics
