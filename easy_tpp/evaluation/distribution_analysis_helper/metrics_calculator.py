"""
Metrics Calculator for Temporal Point Process Analysis.
"""

from typing import Any, Dict

import numpy as np

from easy_tpp.utils import logger


class MetricsCalculatorImpl:
    """Calculates summary metrics from temporal point process data (SRP)."""

    def calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive summary metrics."""
        try:
            metrics = {}

            # Inter-event time metrics
            label_deltas = data.get("label_time_deltas", np.array([]))
            simulated_deltas = data.get("simulated_time_deltas", np.array([]))

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
            label_lengths = data.get("label_sequence_lengths", [])
            simulated_lengths = data.get("simulated_sequence_lengths", [])

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

        except Exception as e:
            logger.error(f"Failed to calculate summary metrics: {str(e)}")
            return {}
