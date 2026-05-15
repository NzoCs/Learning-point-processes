# new_ltpp/models/mixins/training_mixin.py
"""Mixin for PyTorch Lightning training, validation, test, and predict steps."""

from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT

from new_ltpp.evaluation.metrics_helper import MetricsManager
from new_ltpp.shared_types import Batch, OneStepPred

from new_ltpp.simulation.simulator import Simulator
from new_ltpp.configs.model_config import ModelConfig

from .prediction import PredictionMixin


class TrainingMixin(PredictionMixin):
    """Mixin providing Lightning training/validation/test/predict step implementations.

    Requires:
        - self.loglike_loss (from BaseMixin abstract contract)
        - self.predict_one_step_at_every_event (from PredictionMixin)
        - self.num_event_types (set by Model/NeuralModel)
        - self._simulator (Simulator, injected by base_model.Model.__init__)
    """

    eps: float = torch.finfo(torch.float32).eps

    def __init__(self, **kwargs):
        model_config = kwargs["model_config"]
        model_config = cast(
            ModelConfig, model_config
        )  # For type checking; not used at runtime
        num_samples = model_config.thinning_config.num_sample
        super().__init__(num_samples=num_samples, **kwargs)

        # --- Simulator (for prediction and statistics) ---
        self._simulator: Optional[Simulator] = (
            None  # Optional[Simulator] - Injecté par PredictionStatsCallback
        )

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Batch, batch_idx) -> STEP_OUTPUT:
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
        loss, num_events = self.loglike_loss(batch)
        avg_loss = loss / num_events
        self.log(
            "val_loss",
            avg_loss.item(),
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

        pred = self.predict_one_step_at_every_event(
            time_seqs=batch.time_seqs,
            time_delta_seqs=batch.time_delta_seqs,
            type_seqs=batch.type_seqs,
            valid_event_mask=batch.valid_event_mask,
        )

        batch.time_seqs = batch.time_seqs[:, 1:]
        batch.time_delta_seqs = batch.time_delta_seqs[:, 1:]
        batch.type_seqs = batch.type_seqs[:, 1:]
        batch.valid_event_mask = batch.valid_event_mask[:, 1:]

        self._compute_and_log_metrics(batch, pred, prefix="")
        return avg_loss

    def test_step(self, batch: Batch, batch_idx) -> STEP_OUTPUT:
        loss, num_events = self.loglike_loss(batch)
        avg_loss = loss / num_events
        self.log("test_loss", avg_loss.item(), prog_bar=True, sync_dist=True)

        pred = self.predict_one_step_at_every_event(
            time_seqs=batch.time_seqs,
            time_delta_seqs=batch.time_delta_seqs,
            type_seqs=batch.type_seqs,
            valid_event_mask=batch.valid_event_mask,
        )

        batch.time_seqs = batch.time_seqs[:, 1:]
        batch.time_delta_seqs = batch.time_delta_seqs[:, 1:]
        batch.type_seqs = batch.type_seqs[:, 1:]
        batch.valid_event_mask = batch.valid_event_mask[:, 1:]

        self._compute_and_log_metrics(batch, pred, prefix="")
        # self._handle_test_simulation(batch)
        return avg_loss

    def predict_step(self, batch: Batch, batch_idx, **kwargs) -> STEP_OUTPUT:
        if self._simulator is None:
            raise RuntimeError(
                "Simulator not initialized. PredictionStatsCallback should have injected a Simulator instance."
            )
        simulator: "Simulator" = self._simulator
        sim = simulator.simulate(batch=batch)

        if simulator._statistics_collector is None:
            raise RuntimeError(
                "No statistics collector. Call simulator.init_statistics_collector() first."
            )
        simulator._statistics_collector.update(batch, sim)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_and_log_metrics(
        self, batch: Batch, pred: OneStepPred, prefix: str = ""
    ):
        metrics_helper = MetricsManager(num_event_types=self.num_event_types)
        metrics = metrics_helper.compute_prediction_metrics(batch=batch, pred=pred)
        for key, value in metrics.items():
            if key == "confusion_matrix":
                continue
            self.log(f"{prefix}{key}", value, prog_bar=False, sync_dist=True)
        return metrics

    def _handle_test_simulation(self, batch: Batch):
        if self._simulator is None:
            raise RuntimeError(
                "Simulator not initialized. Should be initialized in a callback."
            )

        simulator: "Simulator" = self._simulator
        sim = simulator.simulate(batch=batch)

        if simulator._statistics_collector is None:
            raise RuntimeError(
                "No statistics collector. Call simulator.init_statistics_collector() first."
            )
        simulator._statistics_collector.update(batch, sim)

    # ------------------------------------------------------------------
    # Log-likelihood utility (shared across model implementations)
    # ------------------------------------------------------------------

    def compute_loglikelihood(
        self,
        time_delta_seq: torch.Tensor,
        lambda_at_event: torch.Tensor,
        lambdas_loss_samples: torch.Tensor,
        seq_mask: torch.Tensor,
        type_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-likelihood following Equation (8) of the NHP paper.

        Args:
            time_delta_seq: [batch_size, seq_len]
            lambda_at_event: [batch_size, seq_len, num_event_types]
            lambdas_loss_samples: [batch_size, seq_len, num_sample, num_event_types]
            seq_mask: [batch_size, seq_len]
            type_seq: [batch_size, seq_len]

        Returns:
            Tuple of (event_ll, non_event_ll, num_events).
        """
        lambda_at_event = lambda_at_event + self.eps
        log_marked_event_lambdas = lambda_at_event.log()

        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),
            target=type_seq,
            ignore_index=self.pad_token_id,
            reduction="none",
        )

        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)
        non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        num_events = event_ll.ne(0.0).sum()

        return event_ll, non_event_ll, num_events
