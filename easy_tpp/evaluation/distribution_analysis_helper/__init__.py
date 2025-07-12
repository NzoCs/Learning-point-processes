"""
Distribution Analysis Helper Package

This package provides modular components for temporal point process distribution analysis,
following SOLID principles. Each module has a single responsibility and clear interfaces.

Modules:
- interfaces: Protocol definitions for data extractors, plot generators, and metrics calculators
- data_extractors: Classes for extracting data from different sources (DataLoader, simulations)
- distribution_analyzer: Statistical analysis and visualization utilities
- base_plot_generator: Abstract base class for plot generators
- plot_generators: Specific plot generator implementations
- metrics_calculator: Implementation for calculating summary metrics
- comparator: Main orchestrator class and factory

Author: Research Team
Date: 2024
"""

# Import all components for easy access
from .interfaces import DataExtractor, PlotGenerator, MetricsCalculator
from .data_extractors import LabelDataExtractor, SimulationDataExtractor
from .distribution_analyzer import DistributionAnalyzer
from .base_plot_generator import BasePlotGenerator
from .plot_generators import (
    InterEventTimePlotGenerator,
    EventTypePlotGenerator,
    SequenceLengthPlotGenerator,
    CrossCorrelationPlotGenerator,
)
from .metrics_calculator import MetricsCalculatorImpl
from .comparator import (
    TemporalPointProcessComparator,
    TemporalPointProcessComparatorFactory,
)

__all__ = [
    # Interfaces
    "DataExtractor",
    "PlotGenerator",
    "MetricsCalculator",
    # Data Extractors
    "LabelDataExtractor",
    "SimulationDataExtractor",
    # Analysis Tools
    "DistributionAnalyzer",
    # Plot Generators
    "BasePlotGenerator",
    "InterEventTimePlotGenerator",
    "EventTypePlotGenerator",
    "SequenceLengthPlotGenerator",
    "CrossCorrelationPlotGenerator",
    # Metrics
    "MetricsCalculatorImpl",
    # Main Classes
    "TemporalPointProcessComparator",
    "TemporalPointProcessComparatorFactory",
]
