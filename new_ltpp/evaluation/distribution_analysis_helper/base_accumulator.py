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

    def __init__(self, max_samples: Optional[int] = None):
        """Initialize the accumulator.
        
        Args:
            max_samples: Maximum number of samples to accumulate (None for unlimited)
        """
        self.max_samples = max_samples
        self._sample_count = 0
        self._is_finalized = False

    @abstractmethod
    def update(self, batch: Batch, simulation: Optional[SimulationResult] = None) -> None:
        """Update accumulator with new batch data.
        
        Args:
            batch: Ground truth batch data
            simulation: Optional simulation results for the batch
        """
        pass

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
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

    def should_continue(self) -> bool:
        """Check if accumulator should continue collecting data."""
        if self.max_samples is None:
            return True
        return self._sample_count < self.max_samples
