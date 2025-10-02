"""
Main MetricsHelper class that orchestrates metrics computation.
"""

from typing import Dict, List, Optional, Union

from .metrics_interfaces import MetricsComputerInterface
from .prediction_metrics_computer import PredictionMetricsComputer
from .shared_types import EvaluationMode, PredictionMetrics, SimulationMetrics
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
        simulation_computer: Optional[MetricsComputerInterface] = None,
        selected_prediction_metrics: Optional[
            List[Union[str, PredictionMetrics]]
        ] = None,
        selected_simulation_metrics: Optional[
            List[Union[str, SimulationMetrics]]
        ] = None,
    ):
        """
        Initialize the metrics helper.

        Args:
            num_event_types: Number of event types in the dataset
            mode: Evaluation mode (prediction or simulation)
            save_dir: Directory for saving results (optional)
            prediction_computer: Custom prediction metrics computer (optional)
            simulation_computer: Custom simulation metrics computer (optional)
            selected_prediction_metrics: List of prediction metrics to compute. If None, compute all.
            selected_simulation_metrics: List of simulation metrics to compute. If None, compute all.
        """
        self.num_event_types = num_event_types
        self.mode = mode
        self.save_dir = save_dir

        # Use dependency injection with default implementations
        self._prediction_computer = prediction_computer or PredictionMetricsComputer(
            num_event_types, selected_metrics=selected_prediction_metrics
        )
        self._simulation_computer = simulation_computer or SimulationMetricsComputer(
            num_event_types, selected_metrics=selected_simulation_metrics
        )

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

    def compute_all_time_metrics(self, batch, pred) -> Dict[str, float]:
        """
        Compute all time-related metrics using the prediction computer.

        Args:
            batch: Input batch data
            pred: Model predictions

        Returns:
            Dictionary of computed time metrics
        """
        if self.mode != EvaluationMode.PREDICTION:
            raise ValueError("Time metrics are only available in prediction mode")

        # Type check to ensure we have the right computer type
        if hasattr(self._prediction_computer, "compute_all_time_metrics"):
            return self._prediction_computer.compute_all_time_metrics(batch, pred)
        else:
            raise AttributeError(
                "Prediction computer does not support time-only metrics computation"
            )

    def compute_all_type_metrics(self, batch, pred) -> Dict[str, float]:
        """
        Compute all type-related metrics using the prediction computer.

        Args:
            batch: Input batch data
            pred: Model predictions

        Returns:
            Dictionary of computed type metrics
        """
        if self.mode != EvaluationMode.PREDICTION:
            raise ValueError("Type metrics are only available in prediction mode")

        # Type check to ensure we have the right computer type
        if hasattr(self._prediction_computer, "compute_all_type_metrics"):
            return self._prediction_computer.compute_all_type_metrics(batch, pred)
        else:
            raise AttributeError(
                "Prediction computer does not support type-only metrics computation"
            )

    def compute_all_simulation_time_metrics(self, batch, pred) -> Dict[str, float]:
        """
        Compute all time-related simulation metrics using the simulation computer.

        Args:
            batch: Input batch data
            pred: Model predictions

        Returns:
            Dictionary of computed simulation time metrics
        """
        if self.mode != EvaluationMode.SIMULATION:
            raise ValueError(
                "Simulation time metrics are only available in simulation mode"
            )

        # Type check to ensure we have the right computer type
        if hasattr(self._simulation_computer, "compute_all_time_metrics"):
            return self._simulation_computer.compute_all_time_metrics(batch, pred)
        else:
            raise AttributeError(
                "Simulation computer does not support time-only metrics computation"
            )

    def compute_all_simulation_type_metrics(self, batch, pred) -> Dict[str, float]:
        """
        Compute all type-related simulation metrics using the simulation computer.

        Args:
            batch: Input batch data
            pred: Model predictions

        Returns:
            Dictionary of computed simulation type metrics
        """
        if self.mode != EvaluationMode.SIMULATION:
            raise ValueError(
                "Simulation type metrics are only available in simulation mode"
            )

        # Type check to ensure we have the right computer type
        if hasattr(self._simulation_computer, "compute_all_type_metrics"):
            return self._simulation_computer.compute_all_type_metrics(batch, pred)
        else:
            raise AttributeError(
                "Simulation computer does not support type-only metrics computation"
            )

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
