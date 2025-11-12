"""
Interfaces for metrics computation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch

from .shared_types import (
    MaskedValues,
    SimulationTimeValues,
    SimulationTypeValues,
    TimeValues,
    TypeValues,
)


class MetricsComputerInterface(ABC):
    """
    Interface for metrics computation classes.

    This interface follows the Interface Segregation Principle (ISP)
    by defining a focused contract for metrics computation.
    """

    @abstractmethod
    def compute_metrics(self, batch: Any, pred: Any) -> Dict[str, float]:
        """
        Compute metrics from batch data and predictions.

        Args:
            batch: Input batch data
            pred: Model predictions

        Returns:
            Dictionary of computed metrics
        """
        pass

    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics for this computer.

        Returns:
            List of metric names
        """
        pass

class TimeMetricsComputerInterface(ABC):
    """Interface for time-related metrics computation."""

    @abstractmethod
    def compute_time_metrics(self, batch: Any, pred: Any) -> Dict[str, float]:
        """
        Compute time-related metrics from batch data and predictions.

        Args:
            batch: Input batch data
            pred: Model predictions

        Returns:
            Dictionary of computed time-related metrics
        """
        pass

    def compute_metrics(self, batch: Any, pred: Any) -> Dict[str, float]:
        """Compute all metrics (time-related in this case)."""
        return self.compute_time_metrics(batch, pred)
    
class TypeMetricsComputerInterface(ABC):
    """Interface for type-related metrics computation."""

    @abstractmethod
    def compute_type_metrics(self, batch: Any, pred: Any) -> Dict[str, float]:
        """
        Compute type-related metrics from batch data and predictions.

        Args:
            batch: Input batch data
            pred: Model predictions

        Returns:
            Dictionary of computed type-related metrics
        """
        pass

    def compute_metrics(self, batch: Any, pred: Any) -> Dict[str, float]:
        """Compute all metrics (type-related in this case)."""
        return self.compute_type_metrics(batch, pred)

class DataExtractorInterface(ABC):
    """Interface for data extraction operations."""

    @abstractmethod
    def extract_values(self, batch: Any, pred: Any) -> Any:
        """Extract relevant values from batch and predictions."""
        pass


class TimeExtractorInterface(ABC):
    """Interface for time-specific data extraction."""

    @abstractmethod
    def extract_time_values(self, batch: Any, pred: Any) -> TimeValues:
        """Extract time-related values from batch and predictions."""
        pass


class TypeExtractorInterface(ABC):
    """Interface for type-specific data extraction."""

    @abstractmethod
    def extract_type_values(self, batch: Any, pred: Any) -> TypeValues:
        """Extract type-related values from batch and predictions."""
        pass


class SimulationTimeExtractorInterface(ABC):
    """Interface for simulation time-specific data extraction."""

    @abstractmethod
    def extract_simulation_time_values(
        self, batch: Any, pred: Any
    ) -> SimulationTimeValues:
        """Extract simulation time-related values from batch and predictions."""
        pass


class SimulationTypeExtractorInterface(ABC):
    """Interface for simulation type-specific data extraction."""

    @abstractmethod
    def extract_simulation_type_values(
        self, batch: Any, pred: Any
    ) -> SimulationTypeValues:
        """Extract simulation type-related values from batch and predictions."""
        pass
