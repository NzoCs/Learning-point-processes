"""
Interfaces for metrics computation following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import torch
from .shared_types import MaskedValues


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


class DataExtractorInterface(ABC):
    """Interface for data extraction operations."""
    
    @abstractmethod
    def extract_values(self, batch: Any, pred: Any) -> Any:
        """Extract relevant values from batch and predictions."""
        pass
