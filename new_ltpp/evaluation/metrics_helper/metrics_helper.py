"""
Main MetricsHelper class that orchestrates metrics computation.
"""

from typing import Dict, List, Optional, Union

import torch

from new_ltpp.shared_types import Batch, OneStepPrediction, SimulationResult

from .predictions_metrics.computer import PredictionMetricsComputer
from .predictions_metrics.pred_types import PredictionMetrics
from .simulation_metrics.simul_types import SimulationMetrics
from .simulation_metrics.computer import SimulationMetricsComputer


class MetricsHelper:
    """
    Main metrics computation orchestrator for TPP evaluation.
    
    Simple interface with three main methods:
    - compute_prediction_metrics: for prediction evaluation
    - compute_simulation_metrics: for simulation evaluation  
    - compute_all_metrics: runs both prediction and simulation metrics
    """

    def __init__(self, num_event_types: int, save_dir: Optional[str] = None):
        """
        Initialize the metrics helper.

        Args:
            num_event_types: Number of event types in the dataset
            save_dir: Directory for saving results (optional)
        """
        self.num_event_types = num_event_types
        self.save_dir = save_dir

        # Initialize computers with default settings
        self._prediction_computer = PredictionMetricsComputer(num_event_types)
        self._simulation_computer = SimulationMetricsComputer(num_event_types)

    def compute_prediction_metrics(
        self,
        batch: Batch,
        pred: OneStepPrediction,
        metrics: Optional[List[Union[str, PredictionMetrics]]] = None,
    ) -> Dict[str, float]:
        """
        Compute prediction metrics (time and type prediction evaluation).

        Args:
            batch: Batch object containing ground truth sequences
            pred: OneStepPrediction with dtime_predict and type_predict
            metrics: List of metrics to compute. If None, compute all available metrics.
                     Can be strings or PredictionMetrics enum values.

        Returns:
            Dictionary of computed prediction metrics
        """
        if metrics is not None:
            # Create a computer with specific metrics
            metrics_computer = PredictionMetricsComputer(
                self.num_event_types, selected_metrics=metrics
            )
            return metrics_computer.compute_metrics(batch, pred)
        else:
            # Use default computer
            return self._prediction_computer.compute_metrics(batch, pred)

    def compute_prediction_time_metrics(
        self, batch: Batch, pred_time_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute only time-related prediction metrics.

        Args:
            batch: Batch object containing ground truth sequences
            pred_time_tensor: Tensor of predicted time deltas

        Returns:
            Dictionary of computed time metrics (time_rmse, time_mae)
        """
        return self._prediction_computer.compute_all_time_metrics(batch, pred_time_tensor)

    def compute_prediction_type_metrics(
        self, batch: Batch, pred_type_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute only type-related prediction metrics.

        Args:
            batch: Batch object containing ground truth sequences
            pred_type_tensor: Tensor of predicted event types

        Returns:
            Dictionary of computed type metrics (type_accuracy, f1_score, recall, precision, etc.)
        """
        return self._prediction_computer.compute_all_type_metrics(batch, pred_type_tensor)

    def compute_simulation_metrics(
        self,
        batch: Batch,
        sim: SimulationResult,
        metrics: Optional[List[Union[str, SimulationMetrics]]] = None,
    ) -> Dict[str, float]:
        """
        Compute simulation metrics (distribution comparison between real and simulated data).

        Args:
            batch: Batch object containing ground truth sequences
            pred: Simulation with time_seqs and type_seqs
            metrics: List of metrics to compute. If None, compute all available metrics.
                     Can be strings or SimulationMetrics enum values.

        Returns:
            Dictionary of computed simulation metrics
        """
        if metrics is not None:
            # Create a computer with specific metrics
            metrics_computer = SimulationMetricsComputer(
                self.num_event_types, selected_metrics=metrics
            )
            return metrics_computer.compute_metrics(batch, sim)
        else:
            # Use default computer
            return self._simulation_computer.compute_metrics(batch, sim)

    def compute_simulation_time_metrics(
        self, batch: Batch, sim: SimulationResult
    ) -> Dict[str, float]:
        """
        Compute only time-related simulation metrics.

        Args:
            batch: Batch object containing ground truth sequences
            pred: Simulation with time_seqs

        Returns:
            Dictionary of computed time distribution metrics
        """
        return self._simulation_computer.compute_all_time_metrics(batch, sim)

    def compute_simulation_type_metrics(
        self, batch: Batch, sim: SimulationResult
    ) -> Dict[str, float]:
        """
        Compute only type-related simulation metrics.

        Args:
            batch: Batch object containing ground truth sequences
            pred: Simulation with type_seqs

        Returns:
            Dictionary of computed type distribution metrics
        """
        return self._simulation_computer.compute_all_type_metrics(batch, sim)

    def compute_all_metrics(
        self,
        batch: Batch,
        prediction_pred: OneStepPrediction,
        simulation_pred: SimulationResult,
        prediction_metrics: Optional[List[Union[str, PredictionMetrics]]] = None,
        simulation_metrics: Optional[List[Union[str, SimulationMetrics]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute both prediction and simulation metrics.

        Args:
            batch: Batch object containing ground truth sequences
            prediction_pred: OneStepPrediction with prediction outputs
            simulation_pred: Simulation with simulation outputs
            prediction_metrics: List of prediction metrics to compute. If None, compute all.
            simulation_metrics: List of simulation metrics to compute. If None, compute all.

        Returns:
            Dictionary with two keys: 'prediction' and 'simulation', each containing their respective metrics
        """
        return {
            "prediction": self.compute_prediction_metrics(
                batch, prediction_pred, metrics=prediction_metrics
            ),
            "simulation": self.compute_simulation_metrics(
                batch, simulation_pred, metrics=simulation_metrics
            ),
        }
