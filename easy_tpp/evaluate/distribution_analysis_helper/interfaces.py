"""
Interfaces for Temporal Point Process Distribution Comparison

This module defines the Protocol interfaces following the Interface Segregation Principle (ISP).
These interfaces ensure that classes only depend on the methods they actually use.

Author: Research Team
Date: 2024
"""

from typing import Dict, List, Any, Protocol
import numpy as np


class DataExtractor(Protocol):
    """Interface for data extraction from different sources."""
    
    def extract_time_deltas(self) -> np.ndarray:
        """Extract time delta sequences."""
        ...
    
    def extract_event_types(self) -> np.ndarray:
        """Extract event type sequences."""
        ...
    
    def extract_sequence_lengths(self) -> List[int]:
        """Extract sequence length information."""
        ...


class PlotGenerator(Protocol):
    """Interface for generating different types of plots."""
    
    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        """Generate and save a plot."""
        ...


class MetricsCalculator(Protocol):
    """Interface for calculating different types of metrics."""
    
    def calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate relevant metrics from data."""
        ...
