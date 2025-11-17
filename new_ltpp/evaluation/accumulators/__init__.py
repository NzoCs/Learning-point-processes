"""
Distribution Analysis Helper Package

This package provides batch-based statistical collection for temporal point process
evaluation during prediction phase.

New Architecture:
- base_accumulator: Abstract base class for all accumulators
- time_accumulator: Inter-event time statistics accumulator
- event_type_accumulator: Event type distribution accumulator
- sequence_length_accumulator: Sequence length statistics accumulator
- moment_accumulator: Statistical moments (mean, variance, skewness, kurtosis) accumulator
- batch_statistics_collector: Main orchestrator for batch-by-batch statistics collection
- distribution_analyzer: Statistical analysis and visualization utilities
- plot_generators: Plot generation implementations
- metrics_calculator: Summary metrics computation

Usage:
    # Initialize collector before prediction
    collector = BatchStatisticsCollector(
        num_event_types=10,
        output_dir="./output",
    )
    
    # In predict_step (for each batch)
    collector.update_batch(batch, simulation)
    
    # After prediction loop
    results = collector.finalize_and_save()

Author: Research Team
Date: 2025
"""

from .base_accumulator import BaseAccumulator
from .base_plot_generator import BasePlotGenerator
from .batch_statistics_collector import BatchStatisticsCollector
from .event_type_accumulator import EventTypeAccumulator
from .metrics_calculator import MetricsCalculatorImpl
from .moment_accumulator import MomentAccumulator
from .plot_generators import (
    CrossCorrelationPlotGenerator,
    EventTypePlotGenerator,
    InterEventTimePlotGenerator,
    SequenceLengthPlotGenerator,
)
from .sequence_length_accumulator import SequenceLengthAccumulator
from .time_accumulator import InterEventTimeAccumulator
from .acc_types import (
    AllStatistics,
    EventTypeStatistics,
    FinalResult,
    MetricsData,
    MomentStatistics,
    PlotData,
    SequenceLengthStatistics,
    TimeStatistics,
)

__all__ = [
    # Main Class (primary interface)
    "BatchStatisticsCollector",
    # Accumulators (base and specific)
    "BaseAccumulator",
    "InterEventTimeAccumulator",
    "EventTypeAccumulator",
    "SequenceLengthAccumulator",
    "MomentAccumulator",
    # Plot Generators
    "BasePlotGenerator",
    "InterEventTimePlotGenerator",
    "EventTypePlotGenerator",
    "SequenceLengthPlotGenerator",
    "CrossCorrelationPlotGenerator",
    # Metrics
    "MetricsCalculatorImpl",
    # Types
    "AllStatistics",
    "TimeStatistics",
    "EventTypeStatistics",
    "SequenceLengthStatistics",
    "MomentStatistics",
    "FinalResult",
    "PlotData",
    "MetricsData",
]
