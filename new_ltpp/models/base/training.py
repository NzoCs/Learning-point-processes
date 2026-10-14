# new_ltpp/models/mixins/training_mixin.py
"""Mixin for PyTorch Lightning training, validation, test, and predict steps."""

from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT

from new_ltpp.evaluation.metrics_helper import MetricsManager
from new_ltpp.models.simulation.simulator import Simulator
from new_ltpp.shared_types import Batch, OneStepPred, uses_stopped_event_budget

from .prediction import PredictionMixin


class TrainingStepsMixin:
    """Mixin providing Lightning training/validation/test/predict step implementations.

    Requires:
        - self.loglike_loss (from BaseMixin abstract contract)
        - self.predict_one_step_at_every_event (from PredictionMixin)
        - self.num_event_types (set by Model/NeuralModel)
        - self._simulator (Simulator, injected by base_model.Model.__init__)
    """

    eps: float = torch.finfo(torch.float32).eps

    def ordered_valid_batch(self, batch: Batch):
        """Keep actual event order, placing masked positions after the history.

        The first valid event is conditioning history, regardless of pad side.
        This does not sort/reorder real events or mutate the caller's batch.
        """
        mask = batch.valid_event_mask
        counts = mask.sum(-1)
        expected = (
            torch.arange(mask.size(1), device=mask.device)[None, :] < counts[:, None]
        )
        if torch.equal(mask, expected):
            return batch
        indices = torch.argsort((~mask).long(), dim=1, stable=True)
        return Batch(
            batch.time_seqs.gather(1, indices).masked_fill(~expected, 0),
            batch.time_delta_seqs.gather(1, indices).masked_fill(~expected, 0),
            batch.type_seqs.gather(1, indices).masked_fill(
                ~expected, self.pad_token_id
            ),
            expected,
            sequence_ids=batch.sequence_ids,
            no_next_event_mask=batch.no_next_event_mask,
        )

    def terminal_loglikelihood_cost(self, batch: Batch):
        """Known permanent ends contribute -log P(no next event) = H(infinity).

        Unknown status and quota/window censoring add no permanent-end term.
        Like the transition likelihood, this conditions on the first event.
        """
        flags = batch.no_next_event_mask
        if flags is None:
            return batch.time_seqs.new_zeros(())
        if flags.dtype != torch.bool or flags.shape != batch.time_seqs.shape[:1]:
            raise ValueError(
                "Terminal likelihood requires one boolean flag per sequence"
            )
        if not flags.any():
            return batch.time_seqs.new_zeros(())
        batch = self.ordered_valid_batch(batch)
        mask = batch.valid_event_mask
        counts = mask.sum(-1)
        expected = (
            torch.arange(mask.size(1), device=mask.device)[None, :] < counts[:, None]
        )
        if not torch.equal(mask, expected) or (counts[flags] == 0).any():
            raise ValueError(
                "Terminal likelihood requires nonempty histories with right padding"
            )
        # Use each actual last event; padded states must never contribute.
        rows = flags.nonzero(as_tuple=True)[0]
        cost = batch.time_seqs.new_zeros(())
        for length in counts[rows].unique().tolist():
            selected = rows[counts[rows] == length]
            _, limit = self.next_event_hazard(
                time_seqs=batch.time_seqs[selected, :length],
                time_delta_seqs=batch.time_delta_seqs[selected, :length],
                type_seqs=batch.type_seqs[selected, :length],
                compute_last_step_only=True,
            )
            cost = cost + limit.sum()
        return cost

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Batch, batch_idx) -> STEP_OUTPUT:
        batch = self.ordered_valid_batch(batch)
        loss, num_events = self.loglike_loss(batch)
        if num_events <= 0:
            raise ValueError(
                "Per-event training loss requires at least one valid scored event"
            )
        if not torch.isfinite(loss):
            raise ValueError(
                "Nonfinite training likelihood; known permanent termination may have "
                "zero probability under the current model. No parameter is forced to change."
            )
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
        batch = self.ordered_valid_batch(batch)
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
        batch = self.ordered_valid_batch(batch)
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
        return avg_loss

    def predict_step(self, batch: Batch, batch_idx, **kwargs) -> STEP_OUTPUT:
        if self._simulator is None:
            raise RuntimeError(
                "Simulator not initialized. PredictionStatsCallback should have injected a Simulator instance."
            )
        simulator: "Simulator" = self._simulator
        if simulator._statistics_collector is None:
            raise RuntimeError(
                "No statistics collector. Call simulator.init_statistics_collector() first."
            )
        conditional = simulator.conditional_evaluation
        from new_ltpp.evaluation.statistical_testing.statistical_tests.stopped_mmd_test import (
            observe_event_budget,
        )

        if conditional is None:
            sim = self.simulate(
                batch=batch, num_events_to_simulate=batch.time_seqs.size(1)
            )
            compared_batch = (
                observe_event_budget(batch, batch.time_seqs.size(1))
                if uses_stopped_event_budget(self.__class__.__name__)
                else batch
            )
            histories = None
        else:
            from new_ltpp.evaluation.statistical_testing.statistical_tests import (
                split_history_future,
            )

            if batch.sequence_ids is None:
                raise ValueError(
                    "Conditional prediction requires sequence identifiers from the loader"
                )
            histories, compared_batch = split_history_future(
                batch,
                conditional.history_events,
                conditional.future_events,
                allow_stopped=uses_stopped_event_budget(self.__class__.__name__),
                pad_token_id=(
                    self.pad_token_id
                    if uses_stopped_event_budget(self.__class__.__name__)
                    else 0
                ),
            )
            simulator._statistics_collector.validate_conditional_ids(
                batch.sequence_ids, len(batch.time_seqs)
            )
            sim = self.simulate(
                histories, num_events_to_simulate=conditional.future_events
            )
            sim.sequence_ids = list(batch.sequence_ids)

        # Update statistics
        if histories is None:
            simulator._statistics_collector.update(compared_batch, sim)
        else:
            simulator._statistics_collector.update(
                compared_batch, sim, histories=histories, pair_ids=batch.sequence_ids
            )

        # Accumulate results for saving (via SimulationIOManager)
        if self._io_manager is not None:
            self._io_manager.update(sim)

        simulator.last_result = sim
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_and_log_metrics(
        self, batch: Batch, pred: OneStepPred, prefix: str = ""
    ):
        metrics_helper = MetricsManager(num_event_types=self.num_event_types)
        metrics = metrics_helper.compute_prediction_metrics(batch=batch, pred=pred)
        if "no_event_probability" in pred:
            metrics["no_event_probability"] = (
                pred["no_event_probability"][batch.valid_event_mask].mean().item()
            )
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
        sim = self.simulate(batch=batch, num_events_to_simulate=batch.time_seqs.size(1))

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
        event_ll = event_ll * seq_mask

        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)
        non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        num_events = seq_mask.sum()

        return event_ll, non_event_ll, num_events


class TrainingMixin(TrainingStepsMixin, PredictionMixin):
    """Neural model adapter retaining the established prediction and training path."""

    def __init__(self, **kwargs):
        super().__init__(
            num_samples=kwargs["model_config"].thinning_config.num_sample, **kwargs
        )
