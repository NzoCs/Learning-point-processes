"""
Base Plot Generator for Temporal Point Process Analysis
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePlotGenerator(ABC):
    """Abstract base class for plot generators (OCP)."""
    
    @abstractmethod
    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        """Generate and save a plot."""
        pass
