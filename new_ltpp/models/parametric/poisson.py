"""Homogeneous marked Poisson process with direct vectorized device-side sampling."""

import torch
from torch import nn

from new_ltpp.models.base.parametric import ParametricModel
from new_ltpp.shared_types import Batch


class Poisson(ParametricModel):
    def __init__(self, rates=None, **kwargs):
        super().__init__(**kwargs)
        rates = (
            torch.ones(self.num_event_types)
            if rates is None
            else torch.as_tensor(rates, dtype=torch.float32)
        )
        if (
            rates.shape != (self.num_event_types,)
            or not torch.isfinite(rates).all()
            or (rates <= 0).any()
        ):
            raise ValueError(
                "Poisson rates must be finite, positive and match the event types"
            )
        self.log_rates = nn.Parameter(rates.log())

    @property
    def rates(self):
        return self.log_rates.exp()

    def continuation_intensities(self, batch, deltas):
        return self.rates.expand(*deltas.shape, self.num_event_types)

    def compute_intensities_at_sample_dtimes(
        self, *, time_seqs, sample_dtimes=None, compute_last_step_only=False, **kwargs
    ):
        if sample_dtimes is None:
            sample_dtimes = time_seqs.unsqueeze(-1)
        if compute_last_step_only:
            sample_dtimes = sample_dtimes[:, -1:]
        return self.rates.expand(*sample_dtimes.shape, self.num_event_types)

    def loglike_loss(self, batch):
        # First valid event conditions the observed trajectory; score subsequent events.
        mask = batch.valid_event_mask.clone()
        first = mask.long().cumsum(1) == 1
        targets = mask & ~first
        counts = torch.zeros(
            self.num_event_types, device=self.device, dtype=self.rates.dtype
        )
        counts.scatter_add_(
            0,
            batch.type_seqs.masked_fill(~targets, 0).flatten().long(),
            targets.flatten().to(counts.dtype),
        )
        end = batch.time_seqs.masked_fill(~mask, 0).max(1).values
        start = batch.time_seqs.masked_fill(~mask, float("inf")).min(1).values
        duration = torch.where(mask.any(1), end - start, 0).sum()
        return self.rates.sum() * duration - (counts * self.log_rates).sum(), int(
            targets.sum()
        )

    def simulate(self, batch, num_events_to_simulate=None):
        count = (
            batch.time_seqs.size(1) or 100
            if num_events_to_simulate is None
            else num_events_to_simulate
        )
        if count < 0:
            raise ValueError("Event count must be nonnegative")
        with torch.no_grad():
            shape = (batch.time_seqs.size(0), count)
            gaps = (
                torch.empty(
                    shape, device=self.device, dtype=self.rates.dtype
                ).exponential_()
                / self.rates.sum()
            )
            marks = (
                torch.multinomial(
                    self.rates.expand(shape[0], -1), count, replacement=True
                )
                if count
                else torch.empty(shape, dtype=torch.long, device=self.device)
            )
            return Batch(
                gaps.cumsum(1),
                gaps,
                marks,
                torch.ones(shape, dtype=torch.bool, device=self.device),
                sequence_ids=batch.sequence_ids,
            )

    def simulate_until(self, num_sequences, horizon):
        if horizon < 0:
            raise ValueError("Horizon must be nonnegative")
        with torch.no_grad():
            counts = torch.poisson(
                (self.rates.sum() * horizon).expand(num_sequences)
            ).long()
            width = int(counts.max()) if num_sequences else 0
            mask = torch.arange(width, device=self.device)[None, :] < counts[:, None]
            times = (
                (torch.rand(num_sequences, width, device=self.device) * horizon)
                .masked_fill(~mask, float("inf"))
                .sort(1)
                .values
            )
            gaps = torch.diff(
                times, prepend=torch.zeros(num_sequences, 1, device=self.device)
            ).masked_fill(~mask, 0)
            marks = (
                torch.multinomial(
                    self.rates.expand(num_sequences, -1), width, replacement=True
                )
                if width
                else torch.empty(num_sequences, 0, device=self.device, dtype=torch.long)
            )
            return Batch(
                times.masked_fill(~mask, 0),
                gaps,
                marks.masked_fill(~mask, self.pad_token_id),
                mask,
            )
