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
        )[
            None, None, :
        ]  # [1,1,K]

        tnorm = tnorm.expand(batch_size, seq_len, self.num_samples_boundary)  # [B,L,K]

        # intensities: [B,L,K,num_events]
        intens = intensity_fn(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            sample_dtimes=tnorm,
            compute_last_step_only=compute_last_step_only,
        )

        # Total intensity across marks → envelope M(t)
        # [B,L,K]
        intens_total = intens.sum(-1)

        # Upper bound = max_k M(t_k)
        # [B,L]
        bound = intens_total.max(-1).values * self.over_sample_rate

        return bound

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
        B, L = rate.shape
        u = torch.empty(B, L, num_samples, self.num_exp, device=self.device)
        u.uniform_(0.0, 1.0)
        return u

    # ----------------------------------------------------------------------
    # 4. Vectorized accept step
    # ----------------------------------------------------------------------
    def sample_accept(
        self,
        u: torch.Tensor,
        rate: torch.Tensor,
        intens: torch.Tensor,
        exp_jumps: torch.Tensor,
    ) -> torch.Tensor:
        """
        u:      [B,L,K,E]
        rate:   [B,L]
        intens: [B,L,K,E]
        exp_jumps: [B,L,K,E]
        """
        # criterion = U * λ / λ(t)
        crit = u * rate[..., None, None] / intens

        # accepted where crit < 1
        mask = crit < 1

        # If no position accepted, fallback = dtime_max
        none_accepted = (~mask).all(dim=-1)  # [B,L,K]

        # First accepted along exp dimension E
        idx = mask.float().argmax(dim=-1)  # [B,L,K]

        # Gather the delta
        gathered = torch.gather(exp_jumps, dim=3, index=idx[..., None])  # [B,L,K,1]

        # if none accepted, return 0.0 (no time jump) ??? Have to check what is best here, maybe dtime_max is better
        # We could also filter these later on by masking 0.0 dts an exactly 0.0 dt should happen with prb 0 ??? idk
        # [B,L,K, 1]
        res = torch.where(
            none_accepted[..., None],
            torch.tensor(0.0, device=self.device),
            gathered,
        )

        return res.squeeze(-1)  # [B,L,K]

    # ----------------------------------------------------------------------
    # 5. One thinning step
    # ----------------------------------------------------------------------
    def draw_next_time_one_step(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        intensity_fn: Callable[..., torch.Tensor],
        num_sample: int,
        compute_last_step_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized thinning step"""

        # 1. upper bound M
        M = self.compute_intensity_upper_bound(
            time_seqs, time_delta_seqs, type_seqs, intensity_fn, compute_last_step_only
        )  # [B,L]

        # 2. exp samples
        exp_j = self.sample_exp_distribution(M)  # [B,L,E]
        exp_j = torch.cumsum(exp_j, dim=-1)  # accumulate

        # 3. evaluate intensity at sampled times
        intens = intensity_fn(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            sample_dtimes=exp_j,
            compute_last_step_only=compute_last_step_only,
        )

        intens_total = intens.sum(-1)  # [B,L,E]

        # 4. tile for num_sample (like in thinning.py)
        intens_total = intens_total[:, :, None, :].expand(
            -1, -1, num_sample, -1
        )  # [B,L,num_sample,E]
        exp_j_tiled = exp_j[:, :, None, :].expand(
            -1, -1, num_sample, -1
        )  # [B,L,num_sample,E]

        # 5. uniform
        u = self.sample_uniform(M, num_sample)  # [B,L,num_sample,E]

        # 6. accept
        res = self.sample_accept(u, M, intens_total, exp_j_tiled)  # [B,L,num_sample]

        # uniform weights
        weights = torch.ones_like(res) / num_sample

        return res.clamp(max=1e5), weights
