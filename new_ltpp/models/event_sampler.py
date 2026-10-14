# Vectorized Event Sampler for Multivariate TPP with Thinning
# Compatible with intensity_fn(time_seq, time_delta_seq, event_seq, dtime, ...)
# Includes: vectorized Exp sampling, vectorized uniform draws, stable accept step.

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn


class SamplingSafetyError(RuntimeError):
    """Generation could not continue safely; no fabricated event is returned."""

    def __init__(
        self, reason, *, rounds, pending, accepted, offset, window_start, window_width
    ):
        self.reason = reason
        self.rounds = rounds
        self.pending_count = int(pending.sum())
        self.state = {
            "pending": pending.detach().clone(),
            "accepted_dtimes": accepted.detach().clone(),
            "offset": offset.detach().clone(),
            "window_start": window_start.detach().clone(),
            "window_width": window_width.detach().clone(),
        }
        super().__init__(
            f"Neural event sampling stopped safely: {reason}; rounds={rounds}, pending={self.pending_count}. No unaccepted event was returned."
        )


class EventSampler(nn.Module):
    def __init__(
        self,
        num_exp: int,
        over_sample_rate: float,
        num_samples_boundary: int,
        dtime_max: float,
        device: torch.device,
        max_sampling_rounds: int = 10000,
    ):
        super().__init__()
        self.num_exp = num_exp
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.device = device
        if max_sampling_rounds <= 0 or num_exp <= 0 or num_samples_boundary <= 0:
            raise ValueError("Sampling budgets and sample counts must be positive")
        if not math.isfinite(dtime_max) or dtime_max < 0:
            raise ValueError("Initial sampling window must be finite and nonnegative")
        self.max_sampling_rounds = max_sampling_rounds

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
        *,
        window_start: torch.Tensor | None = None,
        window_width: torch.Tensor | None = None,
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
        if compute_last_step_only and seq_len == 0:
            seq_len = 1  # One initial interval before the first event.

        fractions = torch.linspace(
            0.0, 1.0, self.num_samples_boundary, device=self.device
        )[None, None, :]
        if window_start is None:
            window_start = torch.zeros(batch_size, 1, device=self.device)
        if window_width is None:
            window_width = torch.full_like(window_start, self.dtime_max or 1.0)
        tnorm = window_start.unsqueeze(-1) + window_width.unsqueeze(-1) * fractions

        tnorm = tnorm.expand(batch_size, seq_len, self.num_samples_boundary)  # [B,L,K]

        # intensities: [B,L,K,num_events]
        intens = intensity_fn(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            valid_event_mask=valid_event_mask,
            sample_dtimes=tnorm.to(time_delta_seqs.dtype),
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
    @torch.no_grad()
    def draw_next_time_one_step(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
        intensity_fn: Callable[..., torch.Tensor],
        num_sample: int,
        compute_last_step_only: bool = False,
        conditional_on_event: bool = False,
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

        if num_sample <= 0:
            raise ValueError("num_sample must be positive")
        owner = getattr(intensity_fn, "__self__", None)
        hazard_factory = getattr(owner, "next_event_hazard", None)
        if callable(hazard_factory):
            from new_ltpp.models.hazard_sampling import invert_hazard

            hazard, limit = hazard_factory(
                time_seqs=time_seqs,
                time_delta_seqs=time_delta_seqs,
                type_seqs=type_seqs,
                valid_event_mask=valid_event_mask,
                compute_last_step_only=compute_last_step_only,
            )
            delays = invert_hazard(
                hazard,
                limit,
                num_sample,
                conditional=conditional_on_event,
                max_rounds=self.max_sampling_rounds,
            )
            converted = delays.to(time_seqs.dtype)
            if (torch.isfinite(delays) & ~torch.isfinite(converted)).any():
                raise RuntimeError(
                    "Finite waiting time exceeds output floating-point range"
                )
            return converted, torch.ones_like(converted) / num_sample
        # Grid envelopes are local to finite windows, never global time caps.
        upper_bound = self.compute_intensity_upper_bound(
            time_seqs,
            time_delta_seqs,
            type_seqs,
            valid_event_mask,
            intensity_fn,
            compute_last_step_only,
        )  # [B, L_out]

        B, L_out = upper_bound.shape

        unaccepted_mask = torch.ones(
            B, L_out, num_sample, dtype=torch.bool, device=self.device
        )
        accepted_dtimes = torch.zeros(B, L_out, num_sample, device=self.device)
        # Many small rejected jumps can otherwise round back to the float32
        # offset. Accumulate the clock in float64, evaluate the model in its dtype.
        current_offset = torch.zeros(B, L_out, device=self.device, dtype=torch.float64)
        window_start = torch.zeros_like(current_offset)
        window_width = torch.full_like(current_offset, self.dtime_max or 1.0)
        iters = 0

        def stop(reason):
            raise SamplingSafetyError(
                reason,
                rounds=iters,
                pending=unaccepted_mask,
                accepted=accepted_dtimes,
                offset=current_offset,
                window_start=window_start,
                window_width=window_width,
            )

        while unaccepted_mask.any():
            active_prefixes = unaccepted_mask.any(-1)
            if iters >= self.max_sampling_rounds:
                stop("proposal budget exhausted")
            if (
                (~torch.isfinite(upper_bound) | (upper_bound <= 0)) & active_prefixes
            ).any():
                stop("nonfinite or nonpositive intensity envelope")
            window_end = window_start + window_width
            if (
                (~torch.isfinite(window_end) | (window_end <= window_start))
                & active_prefixes
            ).any():
                stop(
                    "sampling window cannot advance in the current floating-point precision"
                )
            # 2. exp samples
            exp_j = self.sample_exp_distribution(upper_bound).double()  # [B, L_out, E]
            exp_j_cum = torch.cumsum(exp_j, dim=-1) + current_offset.unsqueeze(
                -1
            )  # [B, L_out, E]
            if ((~torch.isfinite(exp_j_cum)).any(-1) & active_prefixes).any() or (
                (exp_j_cum[..., 0] <= current_offset) & active_prefixes
            ).any():
                stop("nonfinite proposals or floating-point stagnation")
            inside = exp_j_cum < window_end.unsqueeze(-1)
            evaluation_times = torch.minimum(exp_j_cum, window_end.unsqueeze(-1))

            # 3. evaluate intensity at sampled times
            intens = intensity_fn(
                time_seqs=time_seqs,
                time_delta_seqs=time_delta_seqs,
                type_seqs=type_seqs,
                valid_event_mask=valid_event_mask,
                sample_dtimes=evaluation_times.to(time_delta_seqs.dtype),
                compute_last_step_only=compute_last_step_only,
            )

            intens = intens.detach()
            invalid_marks = (~torch.isfinite(intens) | (intens < 0)).any(-1)
            intens_total = intens.sum(-1)  # [B, L_out, E]
            active_queries = unaccepted_mask.any(-1).unsqueeze(-1)
            if ((invalid_marks | ~torch.isfinite(intens_total)) & active_queries).any():
                stop("nonfinite or negative intensity")
            tolerance = 32 * torch.finfo(intens_total.dtype).eps
            if (
                (intens_total > upper_bound.unsqueeze(-1) * (1 + tolerance))
                & inside
                & active_queries
            ).any():
                stop(
                    "intensity exceeds the local envelope; increase boundary resolution or over_sample_rate"
                )

            # 4. Tile for uniform evaluation
            intens_total_tiled = intens_total.unsqueeze(2).expand(
                -1, -1, num_sample, -1
            )  # [B, L_out, num_sample, E]
            exp_j_tiled = exp_j_cum.unsqueeze(2).expand(
                -1, -1, num_sample, -1
            )  # [B, L_out, num_sample, E]

            # 5. uniform
            u = self.sample_uniform(
                upper_bound, num_sample
            )  # [B, L_out, num_sample, E]

            # criterion = U * λ / λ(t)
            mask = (
                u * upper_bound.unsqueeze(-1).unsqueeze(-1) < intens_total_tiled
            ) & inside.unsqueeze(2)

            # 6. Check acceptance
            idx = mask.float().argmax(dim=-1)  # [B, L_out, num_sample]
            accepted_in_this_batch = mask.any(dim=-1)  # [B, L_out, num_sample]

            newly_accepted = unaccepted_mask & accepted_in_this_batch
            gathered_times = torch.gather(
                exp_j_tiled, dim=-1, index=idx.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [B, L_out, num_sample]

            accepted_dtimes = torch.where(
                newly_accepted, gathered_times, accepted_dtimes
            )
            unaccepted_mask = unaccepted_mask & ~newly_accepted

            # Advance the offset for the unaccepted paths to the last evaluated proposal
            advance_window = (exp_j_cum[..., -1] >= window_end) & unaccepted_mask.any(
                -1
            )
            current_offset = torch.minimum(exp_j_cum[..., -1], window_end).detach()
            iters += 1
            if advance_window.any():
                window_start = torch.where(advance_window, window_end, window_start)
                window_width = torch.where(
                    advance_window, window_width * 2, window_width
                )
                upper_bound = self.compute_intensity_upper_bound(
                    time_seqs,
                    time_delta_seqs,
                    type_seqs,
                    valid_event_mask,
                    intensity_fn,
                    compute_last_step_only,
                    window_start=window_start,
                    window_width=window_width,
                ).detach()

        # uniform weights
        converted = accepted_dtimes.to(time_delta_seqs.dtype)
        if (~torch.isfinite(converted)).any():
            stop("finite waiting time exceeds output floating-point range")
        accepted_dtimes = converted
        weights = torch.ones_like(accepted_dtimes) / num_sample

        return accepted_dtimes, weights
