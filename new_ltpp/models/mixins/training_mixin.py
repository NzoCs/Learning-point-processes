# new_ltpp/models/mixins/training_mixin.py
"""Mixin for PyTorch Lightning training, validation, and testing steps."""

from typing import List, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT

from new_ltpp.evaluation.metrics_helper import MetricsManager
from new_ltpp.shared_types import Batch, OneStepPred, SimulationResult
from new_ltpp.utils import logger

from .prediction_mixin import PredictionMixin
from .simulation_mixin import SimulationMixin


class TrainingMixin(PredictionMixin, SimulationMixin):
    """Mixin providing training, validation, and test step implementations.

    Requires: self.loglike_loss, self.predict_one_step_at_every_event,
              self.simulate, self.num_event_types, self.compute_simulation,
              self.max_simul_events, self._statistics_collector
    """

    eps: float = torch.finfo(torch.float32).eps  # Small epsilon for numerical stability

    def __init__(self, pad_token_id: int, **kwargs):
        """Initialize the TrainingMixin.

        Args:
            kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id

    def training_step(self, batch: Batch, batch_idx) -> STEP_OUTPUT:
        """Training step for Lightning.

        Args:
            batch: Batch object containing sequences and masks
            batch_idx: Index of the batch

        Returns:
            STEP_OUTPUT: The output of the training step
        """
        loss, num_events = self.loglike_loss(batch)
        avg_loss = loss / num_events
        self.log(
            "train_loss",
            avg_loss.item(),
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )
        return avg_loss

    def validation_step(self, batch: Batch, batch_idx) -> STEP_OUTPUT:
        """Validation step for Lightning.

        Args:
            batch: Batch object containing sequences and masks
            batch_idx: Index of the batch

        Returns:
            STEP_OUTPUT: The output of the validation step
        """
        # Compute loss on the original batch first
        loss, num_events = self.loglike_loss(batch)
        avg_loss = loss / num_events

        # Log validation loss
        self.log(
            "val_loss",
            avg_loss.item(),
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

        # Compute validation metrics
        pred = self.predict_one_step_at_every_event(
            time_seqs=batch.time_seqs,
            time_delta_seqs=batch.time_delta_seqs,
            type_seqs=batch.type_seqs,
        )

        # Mutate the batch in-place so subsequent operations use sequences starting at the second event
        batch.time_seqs = batch.time_seqs[:, 1:]
        batch.time_delta_seqs = batch.time_delta_seqs[:, 1:]
        batch.type_seqs = batch.type_seqs[:, 1:]
        batch.seq_non_pad_mask = batch.seq_non_pad_mask[:, 1:]

        one_step_metrics = self._compute_and_log_metrics(batch, pred, prefix="")

        return avg_loss

    def test_step(self, batch: Batch, batch_idx) -> STEP_OUTPUT:
        """Test step for Lightning.

        Args:
            batch: Batch object containing sequences and masks
            batch_idx: Index of the batch

        Returns:
            STEP_OUTPUT: The output of the test step
        """
        # Compute loss on the original batch first
        loss, num_events = self.loglike_loss(batch)
        avg_loss = loss / num_events
        self.log("test_loss", avg_loss.item(), prog_bar=True, sync_dist=True)

        # Compute prediction metrics
        pred = self.predict_one_step_at_every_event(
            time_seqs=batch.time_seqs,
            time_delta_seqs=batch.time_delta_seqs,
            type_seqs=batch.type_seqs,
        )

        # Mutate the batch in-place
        batch.time_seqs = batch.time_seqs[:, 1:]
        batch.time_delta_seqs = batch.time_delta_seqs[:, 1:]
        batch.type_seqs = batch.type_seqs[:, 1:]
        batch.seq_non_pad_mask = batch.seq_non_pad_mask[:, 1:]

        self._compute_and_log_metrics(batch, pred, prefix="")

        # Handle simulation if enabled
        self._handle_test_simulation(batch)

        return avg_loss

    def _compute_and_log_metrics(
        self, batch: Batch, pred: OneStepPred, prefix: str = ""
    ):
        """Compute and log prediction metrics.

        Args:
            batch: Batch object
            pred: Prediction results
            prefix: Prefix for metric names (e.g., "sim_")
        """
        metrics_helper = MetricsManager(num_event_types=self.num_event_types)
        metrics = metrics_helper.compute_prediction_metrics(batch=batch, pred=pred)

        for key, value in metrics.items():
            if key == "confusion_matrix":
                continue
            self.log(f"{prefix}{key}", value, prog_bar=False, sync_dist=True)

        return metrics

    def _handle_test_simulation(self, batch: Batch):
        """Handle simulation during test step.

        Args:
            batch: Batch object
        """

        # Run simulation
        sim = self.simulate(batch=batch)

        # Update statistics collector
        if self._statistics_collector is None:
            raise NotImplementedError(
                "No statistics collector initialized. Call 'init_statistics_collector' before testing."
            )

        self._statistics_collector.update_batch(batch, sim)

        # # Compute and log simulation metrics
        # metrics_helper = MetricsManager(num_event_types=self.num_event_types)
        # simulation_metrics = metrics_helper.compute_simulation_metrics(
        #     batch=batch, sim=sim
        # )

        # for key, value in simulation_metrics.items():
        #     if key == "confusion_matrix":
        #         continue
        #     self.log(f"sim_{key}", value, prog_bar=False, sync_dist=True)

    def predict_step(self, batch: Batch, batch_idx, **kwargs) -> STEP_OUTPUT:
        """Prediction step for Lightning.

        Args:
            batch: Batch object containing sequences and masks
            batch_idx: Index of the batch

        Returns:
            STEP_OUTPUT: None (simulations are stored internally)
        """

        # Run simulation
        sim = self.simulate(batch=batch)

        # Update statistics collector
        if self._statistics_collector is None:
            raise NotImplementedError(
                "No statistics collector initialized. Call 'init_statistics_collector' before prediction."
            )

        self._statistics_collector.update_batch(batch, sim)

        return

    def compute_loglikelihood(
        self,
        time_delta_seq: torch.Tensor,
        lambda_at_event: torch.Tensor,
        lambdas_loss_samples: torch.Tensor,
        seq_mask: torch.Tensor,
        type_seq: torch.Tensor,
    ):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids, with padded events having a mark of self.pad_token_id

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """

        # First, add an epsilon to every marked intensity for stability
        lambda_at_event = lambda_at_event + self.eps
        log_marked_event_lambdas = lambda_at_event.log()

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(
                0, 2, 1
            ),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction="none",  # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # Compute non-event LL [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)

        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]

        return event_ll, non_event_ll, num_events
