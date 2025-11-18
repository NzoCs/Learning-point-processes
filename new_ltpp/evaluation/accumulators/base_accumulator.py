"""
Base Accumulator Class for Statistical Collection

This module provides the abstract base class for all statistical accumulators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from new_ltpp.shared_types import Batch, SimulationResult


class BaseAccumulator(ABC):
    """Base class for all statistical accumulators.

    Accumulators are designed to:
    1. Process data batch-by-batch during predict_step
    2. Maintain internal state (accumulated statistics)
    3. Provide final computed results on demand
    """

    def __init__(self, min_sim_events: int = 1):
        """Initialize the accumulator.

        Args:
            min_sim_events: Minimum number of simulated events required per batch
        """
        self.min_sim_events = int(min_sim_events)
        self._sample_count = 0
        self._is_finalized = False

    @abstractmethod
    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Update accumulator with new batch data.

        Args:
            batch: Ground truth batch data
            simulation: Simulation results for the batch (required)
        """
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Compute and return final statistics from accumulated data.

        Returns:
            Dictionary containing computed statistics
        """
        pass

    def reset(self) -> None:
        """Reset accumulator to initial state."""
        self._sample_count = 0
        self._is_finalized = False

    @property
    def sample_count(self) -> int:
        """Return number of samples accumulated."""
        return self._sample_count
