"""
Base Plot Generator Class

Abstract base class for all plot generators.
"""

from typing import Protocol

from .acc_types import PlotData


class IPlotGenerator(Protocol):
    """Protocol for IDE type checking + isinstance() support."""

    def generate_plot(self, data: PlotData, output_path: str) -> None:
        """Generate and save a plot."""
        ...
