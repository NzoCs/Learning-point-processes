"""
Base Plot Generator Class

Abstract base class for all plot generators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePlotGenerator(ABC):
    """Abstract base class for plot generators following Open/Closed Principle."""

    @abstractmethod
    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        """Generate and save a plot.
        
        Args:
            data: Dictionary containing the data to plot
            output_path: Full path where to save the plot
        """
        pass
