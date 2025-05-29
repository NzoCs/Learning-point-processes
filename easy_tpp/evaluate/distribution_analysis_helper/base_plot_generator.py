"""
Base Plot Generator for Temporal Point Process Analysis

This module provides the abstract base class for plot generators following
the Open/Closed Principle (OCP).

Author: Research Team
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePlotGenerator(ABC):
    """Abstract base class for plot generators (OCP)."""
    
    @abstractmethod
    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        """Generate and save a plot."""
        pass
