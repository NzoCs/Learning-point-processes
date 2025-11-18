"""
Type definitions for batch statistics collector.

This module contains TypedDict definitions for type-safe statistics handling.
"""

from __future__ import annotations

from typing import Dict, TypedDict

import numpy as np
import numpy.typing as npt


class PlotData(TypedDict):
    """Type definition for plot generation data."""

    label_time_deltas: npt.NDArray[np.float64]  # Histogram counts
    simulated_time_deltas: npt.NDArray[np.float64]  # Histogram counts
    time_bin_edges: npt.NDArray[np.float64]  # Bin edges for time histogram
    label_event_types: npt.NDArray[np.int64]
    simulated_event_types: npt.NDArray[np.int64]
    label_sequence_lengths: npt.NDArray[np.float64]
    simulated_sequence_lengths: npt.NDArray[np.float64]


class MetricsData(TypedDict):
    """Type definition for metrics calculation data."""

    label_time_deltas: npt.NDArray[np.float64]
    simulated_time_deltas: npt.NDArray[np.float64]
    label_sequence_lengths: npt.NDArray[np.float64]
    simulated_sequence_lengths: npt.NDArray[np.float64]


class TimeStatistics(TypedDict):
    """Type definition for time-related statistics (histogram-based)."""

    gt_time_deltas: npt.NDArray[np.float64]  # Histogram counts
    sim_time_deltas: npt.NDArray[np.float64]  # Histogram counts
    bin_edges: npt.NDArray[np.float64]  # Bin edges for histogram
    gt_count: int
    sim_count: int


class EventTypeStatistics(TypedDict):
    """Type definition for event type statistics."""

    gt_array: npt.NDArray[np.int64]
    sim_array: npt.NDArray[np.int64]
    gt_distribution: npt.NDArray[np.float64]
    sim_distribution: npt.NDArray[np.float64]
    gt_count: int
    sim_count: int


class SequenceLengthStatistics(TypedDict):
    """Type definition for sequence length statistics."""

    gt_array: npt.NDArray[np.float64]
    sim_array: npt.NDArray[np.float64]
    gt_mean: float
    gt_median: float
    sim_mean: float
    sim_median: float
    gt_count: int
    sim_count: int


class CorrelationStatistics(TypedDict):
    """Type definition for autocorrelation statistics."""

    acf_gt_mean: npt.NDArray[np.float64]  # Mean ACF for ground truth (max_lag + 1,)
    acf_sim_mean: npt.NDArray[np.float64]  # Mean ACF for simulation (max_lag + 1,)


class AllStatistics(TypedDict):
    """Type definition for all collected statistics."""

    time: TimeStatistics
    event_type: EventTypeStatistics
    sequence_length: SequenceLengthStatistics
    correlation: CorrelationStatistics


class FinalResult(TypedDict):
    """Type definition for finalize_and_save return value."""

    statistics: AllStatistics
    metrics: Dict[str, float]
    batch_count: int
