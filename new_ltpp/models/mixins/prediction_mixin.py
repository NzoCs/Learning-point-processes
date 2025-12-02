# new_ltpp/models/mixins/prediction_mixin.py
"""Mixin for prediction methods (one-step and multi-step)."""

from typing import Optional, Tuple

import torch

from new_ltpp.shared_types import Batch, OneStepPred

from .base_mixin import BaseMixin


class PredictionMixin(BaseMixin):
    """Mixin providing prediction functionality.

    Requires: self.event_sampler, self.num_sample, self.num_step_gen,
              self.compute_intensities_at_sample_times, self.num_event_types
    """

    def __init__(self, num_sample: int, num_step_gen: int, **kwargs):
        """Initialize the PredictionMixin.

        Args:
            num_sample: Number of samples for one-step prediction
            num_step_gen: Number of steps for multi-step generation
        """
        super().__init__(**kwargs)
        self.num_sample = num_sample
        self.num_step_gen = num_step_gen

    def predict_one_step_at_every_event(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
    ) -> OneStepPred:
        """One-step prediction for every event in the sequence.

        Args:
            batch: Batch object containing sequences and masks

        Returns:
            OneStepPred: Predicted time deltas and event types, [batch_size, seq_len].
        """

        time_delta_seqs = time_delta_seqs[:, :-1]
        type_seqs = type_seqs[:, :-1]
        time_seqs = time_seqs[:, :-1]

        # Draw next time samples
        accepted_dtimes, weights = self._event_sampler.draw_next_time_one_step(
            time_seqs,
            time_delta_seqs,
            type_seqs,
            self.compute_intensities_at_sample_times,
            self.num_sample,
            compute_last_step_only=False,
        )

        # Compute intensities at sampled times
        intensities_at_times = self.compute_intensities_at_sample_times(
            time_seqs = time_seqs, 
            time_delta_seqs = time_delta_seqs,
            type_seqs = type_seqs,
            sample_dtimes = accepted_dtimes,
        )

        # Normalize intensities and compute weighted sum
        intensities_normalized = intensities_at_times / intensities_at_times.sum(
            dim=-1, keepdim=True
        )

        intensities_weighted = torch.einsum(
            "...s,...sm->...m", weights, intensities_normalized
        )

        # Get predictions
        types_pred = torch.argmax(intensities_weighted, dim=-1)
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

        return OneStepPred(dtime_predict=dtimes_pred, type_predict=types_pred)
