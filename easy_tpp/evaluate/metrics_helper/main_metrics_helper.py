"""
Main MetricsHelper class that orchestrates metrics computation.

This class follows SOLID principles:
- Single Responsibility: Only orchestrates metrics computation
- Open/Closed: Open for extension via strategy pattern
- Liskov Substitution: Uses interfaces for substitutability  
- Interface Segregation: Depends only on focused interfaces
- Dependency Inversion: Depends on abstractions, not concretions
"""

from typing import Dict, List, Optional
from .interfaces import MetricsComputerInterface
from .shared_types import EvaluationMode
from .prediction_metrics_computer import PredictionMetricsComputer
from .simulation_metrics_computer import SimulationMetricsComputer


class MetricsHelper:
    """
    Main metrics computation orchestrator.
    
    Uses Strategy Pattern to delegate computation to specialized classes.
    """
    
    def __init__(
        self, 
        num_event_types: int, 
        mode: EvaluationMode = EvaluationMode.PREDICTION,
        save_dir: Optional[str] = None,
        prediction_computer: Optional[MetricsComputerInterface] = None,
        simulation_computer: Optional[MetricsComputerInterface] = None
    ):
        """
        Initialize the metrics helper.
        
        Args:
            num_event_types: Number of event types in the dataset
            mode: Evaluation mode (prediction or simulation)
            save_dir: Directory for saving results (optional)
            prediction_computer: Custom prediction metrics computer (optional)
            simulation_computer: Custom simulation metrics computer (optional)
        """
        self.num_event_types = num_event_types
        self.mode = mode
        self.save_dir = save_dir
        
        # Use dependency injection with default implementations
        self._prediction_computer = prediction_computer or PredictionMetricsComputer(num_event_types)
        self._simulation_computer = simulation_computer or SimulationMetricsComputer(num_event_types)
        
        # Select strategy based on mode
        self._current_computer = self._get_computer_for_mode(mode)
    
    def _get_computer_for_mode(self, mode: EvaluationMode) -> MetricsComputerInterface:
        """Get the appropriate metrics computer for the given mode."""
        if mode == EvaluationMode.PREDICTION:
            return self._prediction_computer
        elif mode == EvaluationMode.SIMULATION:
            return self._simulation_computer
        else:
            raise ValueError(f"Unsupported evaluation mode: {mode}")
    
    def set_mode(self, mode: EvaluationMode) -> None:
        """Change the evaluation mode and switch strategy."""
        self.mode = mode
        self._current_computer = self._get_computer_for_mode(mode)
    
    def set_save_dir(self, save_dir: str) -> None:
        """Set the save directory."""
        self.save_dir = save_dir
    
    def compute_all_metrics(self, batch, pred) -> Dict[str, float]:
        """
        Compute all metrics using the current strategy.
        
        Args:
            batch: Input batch data
            pred: Model predictions
            
        Returns:
            Dictionary of computed metrics
        """
        return self._current_computer.compute_metrics(batch, pred)
    
    def get_available_metrics(self) -> List[str]:
        """Get available metrics for the current mode."""
        return self._current_computer.get_available_metrics()
    
    def set_prediction_computer(self, computer: MetricsComputerInterface) -> None:
        """Set a custom prediction metrics computer."""
        self._prediction_computer = computer
        if self.mode == EvaluationMode.PREDICTION:
            self._current_computer = computer
    
    def set_simulation_computer(self, computer: MetricsComputerInterface) -> None:
        """Set a custom simulation metrics computer."""
        self._simulation_computer = computer
        if self.mode == EvaluationMode.SIMULATION:
            self._current_computer = computer
