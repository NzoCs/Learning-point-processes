# new_ltpp/models/mixins/prediction_mixin.py
"""Mixin for prediction methods (one-step and multi-step)."""

import torch
from typing import Optional

from new_ltpp.models.simulation.simulator import Simulator
from new_ltpp.models.simulation.tpp_io import SimulationIOManager
from new_ltpp.shared_types import SimulationResult, Batch, OneStepPred

from .base_model import NeuralModel


class PredictionMixin(NeuralModel):
    """Mixin providing prediction functionality.

    Requires: self.event_sampler, self.num_sample,
              self.compute_intensities_at_sample_times, self.num_event_types
    """

    def __init__(self, num_samples: int, **kwargs):
        """Initialize the PredictionMixin.

        Args:
            num_sample: Number of samples for one-step prediction
        """
        super().__init__(**kwargs)
        self.num_sample = num_samples

        self._simulator: Optional[Simulator] = (
            None  # Optional[Simulator] - Injecté par PredictionStatsCallback
        )
        self._io_manager: Optional[SimulationIOManager] = (
            None  # Optional[SimulationIOManager] - Injecté par PredictionStatsCallback
        )

    def predict_one_step_at_every_event(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
    ) -> OneStepPred:
        """One-step prediction for every event in the sequence.

        Args:
            time_seqs: Tensor of event times, shape [batch_size, seq_len]
            time_delta_seqs: Tensor of time deltas, shape [batch_size, seq_len]
            type_seqs: Tensor of event types, shape [batch_size, seq_len]
            valid_event_mask: Mask tensor indicating non-padding positions, shape [batch_size, seq_len]

        Returns:
            OneStepPred: Predicted time deltas and event types, [batch_size, seq_len].
        """

        time_delta_seqs = time_delta_seqs[:, :-1]
        type_seqs = type_seqs[:, :-1]
        time_seqs = time_seqs[:, :-1]
        valid_event_mask = valid_event_mask[:, :-1]

        # Draw next time samples
        accepted_dtimes, weights = self.get_event_sampler().draw_next_time_one_step(
            time_seqs,
            time_delta_seqs,
            type_seqs,
            valid_event_mask,
            self.compute_intensities_at_sample_dtimes,
            self.num_sample,
            compute_last_step_only=False,
        )

        # Compute intensities at sampled times
        intensities_at_times = self.compute_intensities_at_sample_dtimes(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            valid_event_mask=valid_event_mask,
            sample_dtimes=accepted_dtimes,
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

    def simulate(
        self,
        batch: Batch,
        max_events: int = 10_000,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> SimulationResult:
        if self._simulator is None:
            raise RuntimeError(
                "Simulator not initialized. Should be initialized in a callback."
            )

        simulator: "Simulator" = self._simulator
        sim = simulator.simulate(
            batch=batch, max_events=max_events, start_time=start_time, end_time=end_time
        )

        return sim

    def _create_empty_batch(
        self, batch_size: int, initial_buffer_size: int = 100
    ) -> Batch:
        device = self.device
        return Batch(
            time_seqs=torch.zeros(
                batch_size, initial_buffer_size, device=device, dtype=torch.float32
            ),
            time_delta_seqs=torch.zeros(
                batch_size, initial_buffer_size, device=device, dtype=torch.float32
            ),
            type_seqs=torch.zeros(
                batch_size, initial_buffer_size, device=device, dtype=torch.long
            ),
            valid_event_mask=torch.ones(
                batch_size, initial_buffer_size, device=device, dtype=torch.bool
            ),
        )

    def simulate_from_scratch(
        self,
        num_sequences: int,
        initial_buffer_size: int = 100,
        max_events: int = 10_000,
        start_time: Optional[float] = 0.0,
        end_time: Optional[float] = 100.0,
    ) -> SimulationResult:
        """Simulate event sequences from scratch (no conditioning).

        Args:
            num_sequences: Number of sequences to simulate (defaults to self.batch_size).
            start_time: Optional start time for the simulation.
            end_time: Optional end time for the simulation.

        Returns:
            SimulationResult (Batch alias) with generated sequences.
        """
        empty_batch = self._create_empty_batch(num_sequences, initial_buffer_size)
        return self.simulate(
            empty_batch, max_events, start_time=start_time, end_time=end_time
        )
