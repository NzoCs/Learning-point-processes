"""Protocol definition for Temporal Point Process models.

This module defines the interface that all TPP models must implement,
providing type safety and clear contracts for model behavior.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Union

import torch
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT

from new_ltpp.configs.model_config import ModelConfig

if TYPE_CHECKING:
    from .base_model import OptimizerConfig
    from new_ltpp.evaluation.accumulators.summary_statistics_accumulator import (
        BatchStatisticsCollector,
    )


from new_ltpp.shared_types import Batch, DataInfo, SimulationResult
from new_ltpp.shared_types import OneStepPred


class ITPPModel(Protocol):
    """Protocol defining the interface for Temporal Point Process models.

    This protocol ensures that all TPP models implement the required methods
    for training, evaluation, and inference. It provides type safety and
    clear documentation of the model interface.

    All models should inherit from the base Model class which implements this protocol.
    """

    # ============================================================
    # Required Attributes
    # ============================================================

    num_event_types: int
    """Number of distinct event types in the dataset."""

    pad_token_id: int
    """Token ID used for padding sequences."""

    output_dir: Path
    """Directory for saving model outputs and artifacts."""

    _statistics_collector: Optional["BatchStatisticsCollector"]
    """Collector for batch statistics."""

    def __init__(
        self,
        *,
        model_config: "ModelConfig",
        data_info: "DataInfo",
        output_dir: Path | str,
        **kwargs,
    ) -> None:
        """Initialize the model with configuration and data information.

        Args:
            model_config: Configuration object containing model hyperparameters
            data_info: Dictionary with dataset statistics (num_event_types, end_time_max, dtime_max, pad_token_id)
            output_dir: Directory to save model outputs
            **kwargs: Additional arguments for specific model implementations
        """
        ...

    # ============================================================
    # Core Model Methods
    # ============================================================

    def loglike_loss(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the negative log-likelihood loss for a batch.

        This is the primary loss function for training TPP models. It should
        compute both the event likelihood (likelihood of observed events) and
        the non-event likelihood (integral term for unobserved events).

        Args:
            batch: Input batch containing:
                - time_seqs: Event timestamps [batch_size, seq_len]
                - time_delta_seqs: Inter-event times [batch_size, seq_len]
                - type_seqs: Event types [batch_size, seq_len]
                - valid_event_mask: Mask for valid events [batch_size, seq_len]

        Returns:
            Tuple of (loss, num_events) where:
                - loss: Scalar tensor with the negative log-likelihood
                - num_events: Number of valid (non-padded) events in the batch

        Example:
            >>> loss, num_events = model.loglike_loss(batch)
            >>> avg_loss = loss / num_events
        """
        ...

    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
    ) -> torch.Tensor:
        """Compute intensity function values at arbitrary sample times.

        This method evaluates the intensity function λ(t) at sampled time points,
        not just at observed event times. It's used for:
        - Computing the non-event likelihood (integral approximation)
        - Thinning algorithm for event generation
        - Visualization of intensity functions

        Args:
            time_seqs: Cumulative event times [batch_size, seq_len]
            time_delta_seqs: Inter-event times [batch_size, seq_len]
            type_seqs: Event type sequences [batch_size, seq_len]
            sample_dtimes: Time deltas to evaluate intensity at
                          [batch_size, seq_len, num_samples]
            compute_last_step_only: If True, only compute for the last sequence position
                                   (optimization for one-step-ahead prediction)

        Returns:
            Intensity values [batch_size, seq_len, num_samples, num_event_types]
            Shape is [batch_size, 1, num_samples, num_event_types] if compute_last_step_only=True

        Example:
            >>> # Sample 10 time points uniformly in each inter-event interval
            >>> sample_dtimes = time_delta_seqs[:, :, None] * torch.rand(batch_size, seq_len, 10)
            >>> intensities = model.compute_intensities_at_sample_times(
            ...     time_seqs=batch.time_seqs,
            ...     time_delta_seqs=batch.time_delta_seqs,
            ...     type_seqs=batch.type_seqs,
            ...     sample_dtimes=sample_dtimes
            ... )
        """
        ...

    # ============================================================
    # Prediction Methods
    # ============================================================

    def predict_one_step_at_every_event(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
    ) -> OneStepPred:
        """Predict next event time and type given history up to each event.

        For each event in the sequence, predict what the next event will be
        based only on the history up to that point. This is used for evaluation
        metrics like time prediction error and type accuracy.

        Args:
            batch: Input batch with event sequences

        Returns:
            Tuple of (predicted_dtimes, predicted_types) where:
                - predicted_dtimes: Next inter-event time predictions [batch_size, seq_len]
                - predicted_types: Next event type predictions [batch_size, seq_len]

        Example:
            >>> dtimes_pred, types_pred = model.predict_one_step_at_every_event(batch)
            >>> # Compare predictions with ground truth
            >>> time_error = torch.abs(dtimes_pred - batch.time_delta_seqs[:, 1:])
            >>> type_accuracy = (types_pred == batch.type_seqs[:, 1:]).float().mean()
        """
        ...

    # ============================================================
    # Simulation Methods
    # ============================================================

    def simulate(
        self,
        batch: Optional[Batch] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        batch_size: Optional[int] = None,
        initial_buffer_size: Optional[int] = None,
    ) -> SimulationResult:
        """Generate synthetic event sequences using the learned model.

        Simulate new event sequences from the model using thinning algorithm.
        The model generates events forward in time from start_time to end_time.

        Args:
            start_time: Starting time for simulation
            end_time: Ending time for simulation
            batch_size: Number of sequences to generate in parallel

        Returns:
            Tuple of (time_seqs, time_delta_seqs, type_seqs) where:
                - time_seqs: Cumulative event times [batch_size, max_len]
                - time_delta_seqs: Inter-event times [batch_size, max_len]
                - type_seqs: Event types [batch_size, max_len]

        Example:
            >>> # Generate 5 sequences over time window [0, 100]
            >>> time_seqs, time_delta_seqs, type_seqs = model.simulate(
            ...     start_time=0.0,
            ...     end_time=100.0,
            ...     batch_size=5
            ... )
        """
        ...

    # ============================================================
    # Visualization Methods
    # ============================================================

    def intensity_graph(
        self,
        save_dir: str,
        *,
        precision: int = 100,
        plot: bool = False,
        save_plot: bool = True,
        save_data: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Generate and visualize intensity curves for the model.

        Args:
            precision: Number of interpolation points between events
            plot: Whether to display the plots
            save_plot: Whether to save plots to disk
            save_data: Whether to save intensity data to disk
            save_dir: Directory for saving outputs

        Returns:
            Tuple of (intensities, time_points, marked_times)
        """
        ...

    # ============================================================
    # Statistics Collection
    # ============================================================

    def init_statistics_collector(self, output_dir: Path | str) -> None:
        """Sets self._statistics_collector for the model.

        Args:
            output_dir: Directory to save collected statistics
        """
        ...

    # ============================================================
    # PyTorch Lightning Methods
    # ============================================================

    @property
    def device(self) -> torch.device:
        """Get the current device of the model.

        This property is managed by PyTorch Lightning and always reflects the
        actual device the model is on (CPU, GPU, etc.). It should not be set
        directly; instead, use PyTorch Lightning's device management.

        Returns:
            Current device of the model
        """
        ...

    def configure_optimizers(self) -> Union["OptimizerConfig", optim.Optimizer]:  # type: ignore[override]
        """Configure optimizer and learning rate scheduler.

        Returns optimizer configuration for PyTorch Lightning training.
        Typically returns Adam optimizer with optional cosine annealing scheduler.

        Returns:
            Optimizer or dict with optimizer and scheduler configuration
        """
        ...

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Perform a single training step.

        Args:
            batch: Training batch
            batch_idx: Index of the batch

        Returns:
            Loss value for this training step
        """
        ...

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Perform a single validation step.

        Args:
            batch: Validation batch
            batch_idx: Index of the batch

        Returns:
            Loss value for this validation step
        """
        ...

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Perform a single test step.

        Args:
            batch: Test batch
            batch_idx: Index of the batch

        Returns:
            Dictionary with test metrics
        """
        ...

    def predict_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Perform a single prediction/simulation step.

        Args:
            batch: Input batch for conditioning
            batch_idx: Index of the batch

        Returns:
            Predictions (handled internally, may return None)
        """
        ...


class INeuralTPPModel(ITPPModel, Protocol):
    """Extended protocol for neural network-based TPP models.

    Adds additional attributes and methods specific to neural architectures
    like transformers, RNNs, etc.
    """

    # ============================================================
    # Neural Architecture Attributes
    # ============================================================

    hidden_size: int
    """Dimensionality of hidden representations."""

    dropout: float
    """Dropout rate for regularization."""
