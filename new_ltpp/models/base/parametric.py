"""Parametric TPPs share Lightning steps, without inheriting neural representations."""

from abc import abstractmethod

import torch

from new_ltpp.shared_types import Batch, OneStepPred

from .base_model import BaseModel
from .training import TrainingStepsMixin


class ParametricModel(TrainingStepsMixin, BaseModel):
    model_family = "parametric"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_sample = kwargs["model_config"].thinning_config.num_sample
        self._simulator = None  # prediction statistics, never the parametric generator
        self._io_manager = None

    @abstractmethod
    def simulate(self, batch: Batch, num_events_to_simulate=None):
        """Generate a continuation using the model's dedicated algorithm."""

    @abstractmethod
    def continuation_intensities(self, batch: Batch, deltas: torch.Tensor):
        """Post-history intensities at relative future times [B,S]."""

    def simulate_from_scratch(self, num_sequences, num_events=100):
        times = torch.empty(num_sequences, 0, device=self.device)
        return self.simulate(
            Batch(times, times.clone(), times.long(), times.bool()), num_events
        )

    def simulate_until(self, num_sequences, horizon):
        times = torch.empty(num_sequences, 0, device=self.device)
        return self.simulate(
            Batch(times, times.clone(), times.long(), times.bool()), horizon=horizon
        )

    def predict_one_step_at_every_event(
        self, time_seqs, time_delta_seqs, type_seqs, valid_event_mask
    ):
        b, length = time_seqs.shape
        positions = length - 1
        if positions < 1:
            return OneStepPred(
                dtime_predict=time_seqs[:, :0], type_predict=type_seqs[:, :0]
            )
        # All causal prefixes are simulated together on the model device.
        masks = valid_event_mask[:, None, :] & (
            torch.arange(length, device=self.device)[None, None, :]
            <= torch.arange(positions, device=self.device)[None, :, None]
        )

        def expand(t):
            return (
                t[:, None, :]
                .expand(b, positions, length)
                .reshape(b * positions, length)
                .repeat_interleave(self.num_sample, dim=0)
            )

        prefixes = Batch(
            expand(time_seqs),
            expand(time_delta_seqs),
            expand(type_seqs),
            masks.reshape(b * positions, length).repeat_interleave(
                self.num_sample, dim=0
            ),
        )
        draws = self.simulate(prefixes, 1)
        deltas = draws.time_delta_seqs
        intensity = self.continuation_intensities(prefixes, deltas)
        probabilities = intensity / intensity.sum(-1, keepdim=True)
        probabilities = probabilities.reshape(
            b, positions, self.num_sample, self.num_event_types
        ).mean(2)
        return OneStepPred(
            dtime_predict=deltas.reshape(b, positions, self.num_sample).mean(-1),
            type_predict=probabilities.argmax(-1),
        )
