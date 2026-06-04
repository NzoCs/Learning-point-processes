# Vectorized Event Sampler for Multivariate TPP with Thinning
# Compatible with intensity_fn(time_seq, time_delta_seq, event_seq, dtime, ...)
# Includes: vectorized Exp sampling, vectorized uniform draws, stable accept step.

from typing import Callable, Tuple

import torch
import torch.nn as nn


class EventSampler(nn.Module):
    def __init__(
        self,
        num_exp: int,
        over_sample_rate: float,
        num_samples_boundary: int,
        dtime_max: float,
        device: torch.device,
    ):
        super().__init__()
        self.num_exp = num_exp
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.device = device

    # ----------------------------------------------------------------------
    # 1. Compute intensity upper bound
    # ----------------------------------------------------------------------
    def compute_intensity_upper_bound(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
        intensity_fn: Callable[..., torch.Tensor],
        compute_last_step_only: bool,
    ) -> torch.Tensor:
        """Compute upper bound M(t) for thinning algorithm.
        Args:
            time_seq: [B,L]
            time_delta_seq: [B,L]
            event_seq: [B,L]
            intensity_fn: Callable
            compute_last_step_only: bool
        Returns:
            bound: [B,L]
        """
        batch_size, seq_len = time_seqs.size()

        tnorm = torch.linspace(
            0.0, self.dtime_max, self.num_samples_boundary, device=self.device
        )[None, None, :]  # [1,1,K]

        tnorm = tnorm.expand(batch_size, seq_len, self.num_samples_boundary)  # [B,L,K]

        # intensities: [B,L,K,num_events]
        intens = intensity_fn(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            valid_event_mask=valid_event_mask,
            sample_dtimes=tnorm,
            compute_last_step_only=compute_last_step_only,
        )

        # Total intensity across marks → envelope M(t)
        # [B,L,K]
        intens_total = intens.sum(-1)

        # Upper bound = max_k M(t_k)
        # [B,L]
        bound = intens_total.max(-1).values * self.over_sample_rate

        return bound.clamp(min=1e-7)

    # ----------------------------------------------------------------------
    # 2. Sample exponential jumps
    # ----------------------------------------------------------------------
    def sample_exp_distribution(self, rate: torch.Tensor) -> torch.Tensor:
        """Sample Exp(rate) i.i.d. with vectorization.
        rate: [B,L]
        returns: [B,L,num_exp]
        """
        B, L = rate.shape
        e = torch.empty(B, L, self.num_exp, device=self.device)
        e.exponential_(1.0)  # Exp(1)
        return e / rate[..., None]

    # ----------------------------------------------------------------------
    # 3. Sample uniform numbers for acceptance
    # ----------------------------------------------------------------------
    def sample_uniform(self, rate: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample uniform numbers for acceptance
        rate: [B,L]
        num_samples: int
        returns: [B,L,num_samples,self.num_exp]"""

        B, L = rate.shape
        u = torch.empty(B, L, num_samples, self.num_exp, device=self.device)
        u.uniform_(0.0, 1.0)
        return u

    # ----------------------------------------------------------------------
    # 4. One thinning step (Rejection Sampling Loop)
    # ----------------------------------------------------------------------
    def draw_next_time_one_step(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
        intensity_fn: Callable[..., torch.Tensor],
        num_sample: int,
        compute_last_step_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized thinning step with Rejection Sampling loop
        Args:
            time_seqs: [B,L]
            time_delta_seqs: [B,L]
            type_seqs: [B,L]
            valid_event_mask: [B,L]
            intensity_fn: Callable[..., torch.Tensor]
            num_sample: int
            compute_last_step_only: bool
        Returns:
            accepted_dtimes: [B,L,num_sample]
            weights: [B,L,num_sample]
        """

        # 1. upper bound M
        upper_bound = self.compute_intensity_upper_bound(
            time_seqs,
            time_delta_seqs,
            type_seqs,
            valid_event_mask,
            intensity_fn,
            compute_last_step_only,
        )  # [B, L_out]

        B, L_out = upper_bound.shape

        unaccepted_mask = torch.ones(B, L_out, num_sample, dtype=torch.bool, device=self.device)
        accepted_dtimes = torch.zeros(B, L_out, num_sample, device=self.device)
        current_offset = torch.zeros(B, L_out, device=self.device)

        max_iters = 10
        iters = 0

        while unaccepted_mask.any() and iters < max_iters:
            # 2. exp samples
            exp_j = self.sample_exp_distribution(upper_bound)  # [B, L_out, E]
            exp_j_cum = torch.cumsum(exp_j, dim=-1) + current_offset.unsqueeze(-1)  # [B, L_out, E]

            # 3. evaluate intensity at sampled times
            intens = intensity_fn(
                time_seqs=time_seqs,
                time_delta_seqs=time_delta_seqs,
                type_seqs=type_seqs,
                valid_event_mask=valid_event_mask,
                sample_dtimes=exp_j_cum,
                compute_last_step_only=compute_last_step_only,
            )

            intens_total = intens.sum(-1)  # [B, L_out, E]

            # 4. Tile for uniform evaluation
            intens_total_tiled = intens_total.unsqueeze(2).expand(-1, -1, num_sample, -1)  # [B, L_out, num_sample, E]
            exp_j_tiled = exp_j_cum.unsqueeze(2).expand(-1, -1, num_sample, -1)  # [B, L_out, num_sample, E]

            # 5. uniform
            u = self.sample_uniform(upper_bound, num_sample)  # [B, L_out, num_sample, E]

            # criterion = U * λ / λ(t)
            crit = u * upper_bound.unsqueeze(-1).unsqueeze(-1) / intens_total_tiled  # [B, L_out, num_sample, E]
            mask = crit < 1  # [B, L_out, num_sample, E]

            # 6. Check acceptance
            idx = mask.float().argmax(dim=-1)  # [B, L_out, num_sample]
            accepted_in_this_batch = mask.any(dim=-1)  # [B, L_out, num_sample]

            newly_accepted = unaccepted_mask & accepted_in_this_batch
            gathered_times = torch.gather(exp_j_tiled, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [B, L_out, num_sample]

            accepted_dtimes = torch.where(newly_accepted, gathered_times, accepted_dtimes)
            unaccepted_mask = unaccepted_mask & ~newly_accepted

            # Advance the offset for the unaccepted paths to the last evaluated proposal
            current_offset = exp_j_cum[..., -1]
            iters += 1

        # Fallback for paths that never accepted after max_iters
        if unaccepted_mask.any():
            accepted_dtimes = torch.where(unaccepted_mask, current_offset.unsqueeze(-1), accepted_dtimes)

        # uniform weights
        weights = torch.ones_like(accepted_dtimes) / num_sample

        return accepted_dtimes.clamp(max=self.dtime_max), weights
