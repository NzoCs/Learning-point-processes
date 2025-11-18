from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy

from new_ltpp.evaluation.accumulators.acc_types import (
    AllStatistics,
    CorrelationStatistics,
    SequenceLengthStatistics,
)
from new_ltpp.utils import logger

from .summary_stats_metrics import SummaryStatsMetric


class SummaryStatsHelper:
    """Computes metrics that summarize accumulated statistics."""

    EPSILON: float = float(np.finfo(np.float64).eps)

    def __init__(
        self,
        selected_metrics: Optional[List[Union[str, SummaryStatsMetric]]] = None,
    ) -> None:
        processed_metrics: Set[str]
        if selected_metrics is None:
            processed_metrics = set(self.get_available_metrics())
        else:
            processed_metrics = self._normalize_metric_selection(selected_metrics)
            available = set(self.get_available_metrics())
            invalid = processed_metrics - available
            if invalid:
                logger.warning(
                    "Invalid summary stats metrics requested: %s. Available: %s",
                    invalid,
                    available,
                )
            processed_metrics &= available

        self.selected_metrics: Set[str] = processed_metrics

    def compute_metrics(self, statistics: AllStatistics) -> Dict[str, float]:
        """Compute the configured summary statistics metrics."""
        metrics: Dict[str, float] = {}

        time_stats = statistics["time"]
        event_stats = statistics["event_type"]
        sequence_stats = statistics["sequence_length"]
        correlation_stats = statistics["correlation"]

        gt_time_hist = self._to_float_array(time_stats["gt_time_deltas"])
        sim_time_hist = self._to_float_array(time_stats["sim_time_deltas"])

        gt_event_dist = self._to_float_array(event_stats["gt_distribution"])
        sim_event_dist = self._to_float_array(event_stats["sim_distribution"])

        metric_mapping = self._build_metric_mapping(
            gt_time_hist,
            sim_time_hist,
            gt_event_dist,
            sim_event_dist,
            sequence_stats,
            correlation_stats,
        )

        for metric_name, (func, *args) in metric_mapping.items():
            if metric_name in self.selected_metrics:
                metrics[metric_name] = func(*args)

        return metrics


    @staticmethod
    def get_available_metrics() -> List[str]:
        return [metric.value for metric in SummaryStatsMetric]

    def _normalize_metric_selection(
        self, selections: Iterable[Union[str, SummaryStatsMetric]]
    ) -> Set[str]:
        normalized: Set[str] = set()
        for metric in selections:
            if isinstance(metric, SummaryStatsMetric):
                normalized.add(metric.value)
            else:
                normalized.add(str(metric))
        return normalized

    def _to_float_array(
        self, array: NDArray[Any] | Sequence[Any]
    ) -> NDArray[np.float64]:
        return np.asarray(array, dtype=float)

    def _align_histograms(
        self, gt: np.ndarray, sim: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        max_len = max(gt.size, sim.size)
        if max_len == 0:
            return gt, sim

        aligned_gt = np.zeros(max_len, dtype=float)
        aligned_sim = np.zeros(max_len, dtype=float)
        aligned_gt[: gt.size] = gt
        aligned_sim[: sim.size] = sim
        return aligned_gt, aligned_sim

    def _histogram_distance_l1(self, gt: np.ndarray, sim: np.ndarray) -> float:
        aligned_gt, aligned_sim = self._align_histograms(gt, sim)
        if aligned_gt.size == 0 and aligned_sim.size == 0:
            return 0.0
        return float(np.linalg.norm(aligned_gt - aligned_sim, ord=1))

    def _histogram_distance_l2(self, gt: np.ndarray, sim: np.ndarray) -> float:
        aligned_gt, aligned_sim = self._align_histograms(gt, sim)
        if aligned_gt.size == 0 and aligned_sim.size == 0:
            return 0.0
        return float(np.linalg.norm(aligned_gt - aligned_sim, ord=2))

    def _kl_divergence(self, gt: np.ndarray, sim: np.ndarray) -> float:
        aligned_gt, aligned_sim = self._align_histograms(gt, sim)
        if aligned_gt.size == 0 and aligned_sim.size == 0:
            return float("nan")

        gt_norm = self._safe_normalize(aligned_gt)
        sim_norm = self._safe_normalize(aligned_sim)

        if not np.any(gt_norm) or not np.any(sim_norm):
            return float("nan")

        return float(entropy(gt_norm + self.EPSILON, sim_norm + self.EPSILON))

    def _safe_normalize(self, values: np.ndarray) -> np.ndarray:
        total = float(np.sum(values))
        if total <= 0:
            return np.zeros_like(values)
        return values / total

    @staticmethod
    def _absolute_difference(a: float, b: float) -> float:
        return float(abs(float(a) - float(b)))


    def _build_metric_mapping(
        self,
        gt_time_hist: NDArray[np.float64],
        sim_time_hist: NDArray[np.float64],
        gt_event_dist: NDArray[np.float64],
        sim_event_dist: NDArray[np.float64],
        sequence_stats: SequenceLengthStatistics,
        correlation_stats: CorrelationStatistics,
    ) -> Dict[str, tuple[Callable[..., float], Any, Any]]:
        return {
            SummaryStatsMetric.TIME_HIST_L1.value: (
                self._histogram_distance_l1,
                gt_time_hist,
                sim_time_hist,
            ),
            SummaryStatsMetric.TIME_HIST_L2.value: (
                self._histogram_distance_l2,
                gt_time_hist,
                sim_time_hist,
            ),
            SummaryStatsMetric.TIME_HIST_KL.value: (
                self._kl_divergence,
                gt_time_hist,
                sim_time_hist,
            ),
            SummaryStatsMetric.EVENT_TYPE_HIST_L1.value: (
                self._histogram_distance_l1,
                gt_event_dist,
                sim_event_dist,
            ),
            SummaryStatsMetric.EVENT_TYPE_HIST_L2.value: (
                self._histogram_distance_l2,
                gt_event_dist,
                sim_event_dist,
            ),
            SummaryStatsMetric.EVENT_TYPE_HIST_KL.value: (
                self._kl_divergence,
                gt_event_dist,
                sim_event_dist,
            ),
            SummaryStatsMetric.SEQUENCE_LENGTH_MEAN_DIFF.value: (
                self._absolute_difference,
                sequence_stats["gt_mean"],
                sequence_stats["sim_mean"],
            ),
            SummaryStatsMetric.SEQUENCE_LENGTH_MEDIAN_DIFF.value: (
                self._absolute_difference,
                sequence_stats["gt_median"],
                sequence_stats["sim_median"],
            ),
        }