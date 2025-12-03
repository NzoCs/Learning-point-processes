"""Protocol definition for Temporal Point Process models.

This module defines the interface that all TPP models must implement,
providing type safety and clear contracts for model behavior.
"""

from typing import Protocol, runtime_checkable

import torch
from pathlib import Path

from new_ltpp.shared_types import Batch


@runtime_checkable
class TPPModelProtocol(Protocol):
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
    
    device: torch.device
    """Current device of the model (managed by PyTorch Lightning)."""
    
    output_dir: Path
    """Directory for saving model outputs and artifacts."""
    
    # ============================================================
    # Core Model Methods
    # ============================================================
    
    def loglike_loss(self, batch: Batch) -> tuple[torch.Tensor, int]:
        """Compute the negative log-likelihood loss for a batch.
        
        This is the primary loss function for training TPP models. It should
        compute both the event likelihood (likelihood of observed events) and
        the non-event likelihood (integral term for unobserved events).
        
        Args:
            batch: Input batch containing:
                - time_seqs: Event timestamps [batch_size, seq_len]
                - time_delta_seqs: Inter-event times [batch_size, seq_len]
                - type_seqs: Event types [batch_size, seq_len]
                - seq_non_pad_mask: Mask for valid events [batch_size, seq_len]
        
        Returns:
            Tuple of (loss, num_events) where:
                - loss: Scalar tensor with the negative log-likelihood
                - num_events: Number of valid (non-padded) events in the batch
        
        Example:
            >>> loss, num_events = model.loglike_loss(batch)
            >>> avg_loss = loss / num_events
        """
        ...
    
    def compute_intensities_at_sample_times(
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
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        *,
        start_time: float,
        end_time: float,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        *,
        save_dir: str,
        start_time: float | None = None,
        end_time: float | None = None,
        precision: int = 100,
    ) -> None:
        """Generate and save intensity function visualization.
        
        Creates plots showing the intensity function λ(t) over time for each
        event type. This helps understand the temporal dynamics learned by the model.
        
        Args:
            save_dir: Directory to save the intensity plots
            start_time: Start of time window (uses simulation default if None)
            end_time: End of time window (uses simulation default if None)
            precision: Number of time points to evaluate for smooth curves
        
        Example:
            >>> model.intensity_graph(
            ...     save_dir="./outputs/intensity_plots",
            ...     start_time=0.0,
            ...     end_time=50.0,
            ...     precision=200
            ... )
        """
        ...
    
    # ============================================================
    # Statistics Collection
    # ============================================================
    
    def finalize_statistics(self) -> None:
        """Finalize and save collected statistics from prediction/simulation.
        
        After running predictions or simulations, this method computes aggregate
        statistics and generates comparison plots between ground truth and
        generated sequences. Called automatically at the end of prediction phase.
        
        Example:
            >>> trainer.predict(model, dataloader)
            >>> model.finalize_statistics()  # Saves statistics and plots
        """
        ...
    
    # ============================================================
    # PyTorch Lightning Methods
    # ============================================================
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.
        
        Returns optimizer configuration for PyTorch Lightning training.
        Typically returns Adam optimizer with optional cosine annealing scheduler.
        
        Returns:
            Optimizer or dict with optimizer and scheduler configuration
        """
        ...
    
    def training_step(self, batch: Batch, batch_idx: int):
        """Perform a single training step.
        
        Args:
            batch: Training batch
            batch_idx: Index of the batch
        
        Returns:
            Loss value for this training step
        """
        ...
    
    def validation_step(self, batch: Batch, batch_idx: int):
        """Perform a single validation step.
        
        Args:
            batch: Validation batch
            batch_idx: Index of the batch
        
        Returns:
            Loss value for this validation step
        """
        ...
    
    def test_step(self, batch: Batch, batch_idx: int):
        """Perform a single test step.
        
        Args:
            batch: Test batch
            batch_idx: Index of the batch
        
        Returns:
            Dictionary with test metrics
        """
        ...
    
    def predict_step(self, batch: Batch, batch_idx: int):
        """Perform a single prediction/simulation step.
        
        Args:
            batch: Input batch for conditioning
            batch_idx: Index of the batch
        
        Returns:
            Predictions (handled internally, may return None)
        """
        ...


@runtime_checkable
class NeuralTPPModelProtocol(TPPModelProtocol, Protocol):
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
    
    # ============================================================
    # Neural Network Methods
    # ============================================================
    
    def forward(
        self,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the neural network encoder.
        
        Processes the input sequences through the model's encoder (e.g., Transformer,
        RNN) to produce hidden representations at each time step.
        
        Args:
            time_seqs: Event time sequences [batch_size, seq_len]
            type_seqs: Event type sequences [batch_size, seq_len]
            **kwargs: Additional model-specific arguments (e.g., attention masks)
        
        Returns:
            Hidden representations [batch_size, seq_len, hidden_size]
        
        Example:
            >>> hidden = model.forward(
            ...     time_seqs=batch.time_seqs,
            ...     type_seqs=batch.type_seqs,
            ...     attn_mask=causal_mask
            ... )
        """
        ...


# ============================================================
# Type Guards and Validation
# ============================================================

def is_valid_tpp_model(obj: object) -> bool:
    """Check if an object implements the TPPModelProtocol.
    
    Args:
        obj: Object to check
    
    Returns:
        True if object implements the protocol
    
    Example:
        >>> from new_ltpp.models import THP
        >>> model = THP(...)
        >>> assert is_valid_tpp_model(model)
    """
    return isinstance(obj, TPPModelProtocol)


def is_neural_tpp_model(obj: object) -> bool:
    """Check if an object implements the NeuralTPPModelProtocol.
    
    Args:
        obj: Object to check
    
    Returns:
        True if object implements the neural model protocol
    
    Example:
        >>> model = THP(...)
        >>> assert is_neural_tpp_model(model)
    """
    return isinstance(obj, NeuralTPPModelProtocol)
