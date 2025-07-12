"""
Metrics Helper Package

A modular metrics computation system following SOLID principles.
"""

from .interfaces import (
    MetricsComputerInterface,
    DataExtractorInterface,
    TimeExtractorInterface,
    TypeExtractorInterface,
    SimulationTimeExtractorInterface,
    SimulationTypeExtractorInterface,
)
from .main_metrics_helper import MetricsHelper
from .prediction_metrics_computer import PredictionMetricsComputer
from .simulation_metrics_computer import SimulationMetricsComputer
from .shared_types import (
    EvaluationMode,
    MaskedValues,
    TimeValues,
    TypeValues,
    SimulationTimeValues,
    SimulationTypeValues,
    SimulationValues,
    PredictionMetrics,
    SimulationMetrics,
)

__all__ = [
    "MetricsComputerInterface",
    "DataExtractorInterface",
    "TimeExtractorInterface",
    "TypeExtractorInterface",
    "SimulationTimeExtractorInterface",
    "SimulationTypeExtractorInterface",
    "MetricsHelper",
    "PredictionMetricsComputer",
    "SimulationMetricsComputer",
    "EvaluationMode",
    "MaskedValues",
    "TimeValues",
    "TypeValues",
    "SimulationTimeValues",
    "SimulationTypeValues",
    "SimulationValues",
    "PredictionMetrics",
    "SimulationMetrics",
]
