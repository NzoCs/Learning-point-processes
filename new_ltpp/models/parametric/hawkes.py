"""
PyTorch implementation of the multivariate Hawkes process model.

Used as a parametric benchmark against neural TPP models — supports exact
log-likelihood computation and simulation via Ogata's thinning algorithm.

Intensity:
    λ_k(t) = softplus(μ_k + Σ_{t_i < t} α[k, m_i] · β[k, m_i] · exp(−β[k, m_i] · (t − t_i)))
"""

from new_ltpp.models.base import TrainingMixin

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from new_ltpp.shared_types import Batch, SimulationResult


class Hawkes(TrainingMixin):
    """
    Multivariate Hawkes process with matrix-valued (α, β) parameters.

    Parameters are learnable (nn.Parameter), making this class compatible with
    gradient-based fitting via NLL minimisation.

    Args:
        mu   : Baseline intensities.   Shape (K,)    — learnable.
        alpha: Excitation magnitudes.  Shape (K, K)  — α[k, m] = effect of type-m on type-k.
        beta : Exponential decay rates.Shape (K, K)  — β[k, m] > 0 (enforced via softplus).
    """

    mu: nn.Parameter
    alpha: nn.Parameter
    beta: nn.Parameter

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        mu: Optional[list[float] | torch.Tensor] = None,
        alpha: Optional[list[list[float]] | torch.Tensor] = None,
        beta: Optional[list[list[float]] | torch.Tensor] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        K = (
            self.num_event_types
            if hasattr(self, "num_event_types")
            else (
                len(mu)
                if isinstance(mu, list)
                else (mu.shape[0] if mu is not None else 1)
            )
        )
        if not hasattr(self, "num_event_types"):
            self.num_event_types = K

        self.eps = 1e-5
        dev = getattr(self, "device", torch.device("cpu"))

        def _to_param(x, default, shape, name):
            if x is None:
                x = default
            t = torch.tensor(x, dtype=torch.float32, device=dev).view(shape)
            if t.shape != torch.Size(shape):
                raise ValueError(
                    f"Hawkes: expected {name} of shape {shape}, got {list(t.shape)}"
                )
            return t

        self.mu = nn.Parameter(_to_param(mu, torch.zeros(K), (K,), "mu"))
        self.alpha = nn.Parameter(
            _to_param(alpha, torch.zeros((K, K)), (K, K), "alpha")
        )
        self.beta = nn.Parameter(_to_param(beta, torch.ones((K, K)), (K, K), "beta"))

    # ──────────────────────────────────────────────────────────────────────────
    # Properties — positive-constrained parameters
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def mu_pos(self) -> torch.Tensor:
        return F.softplus(self.mu)

    @property
    def alpha_pos(self) -> torch.Tensor:
        return F.softplus(self.alpha)

    @property
    def beta_pos(self) -> torch.Tensor:
        return F.softplus(self.beta)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _safe_embedding(
        self,
        type_seqs: torch.Tensor,  # [B, L]
        valid_mask: torch.Tensor,  # [B, L] bool
        weight: torch.Tensor,  # [K, K]
    ) -> torch.Tensor:
        """Embedding with padding-safe index clamping."""
        safe = type_seqs.long().clone()
        safe[~valid_mask.bool()] = 0
        out = F.embedding(safe, weight)  # [B, L, K]
        return out * valid_mask.float().unsqueeze(-1)

    def _compute_start_end_time(
        self,
        time_seqs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Default simulation window: [t_last, t_last + (t_last - t_first)]."""
        t_min = time_seqs.clone()
        t_min[~valid_mask] = float("inf")
        t_first = t_min.min(dim=1).values
        t_first[torch.isinf(t_first)] = 0.0

        t_max = time_seqs.clone()
        t_max[~valid_mask] = 0.0
        t_last = t_max.max(dim=1).values

        sim_window = (t_last - t_first).clamp(min=1e-3)
        return t_last, t_last + sim_window

    # ──────────────────────────────────────────────────────────────────────────
    # Core intensity computation
    # ──────────────────────────────────────────────────────────────────────────

    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,  # [B, L]
        type_seqs: torch.Tensor,  # [B, L]
        valid_event_mask: torch.Tensor,  # [B, L]
        sample_dtimes: Optional[torch.Tensor] = None,  # [B, L, S]
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        B, L = time_seqs.shape
        dev = time_seqs.device

        # Elapsed times — [B, L_query, L_src, S] or [B, L_query, L_src, 1]
        base = time_seqs.unsqueeze(2) - time_seqs.unsqueeze(1)  # [B, L, L]
        if sample_dtimes is not None:
            tau = base.unsqueeze(-1) + sample_dtimes.unsqueeze(2)
        else:
            tau = base.unsqueeze(-1)

        tau = tau.abs()

        if compute_last_step_only:
            tau = tau[:, -1:, :, :]

        # Positive-constrained params (computed once here for NLL path)
        mu = self.mu_pos
        alpha = self.alpha_pos
        beta = self.beta_pos

        alpha_src = self._safe_embedding(
            type_seqs, valid_event_mask, alpha.t()
        )  # [B, L, K]
        beta_src = self._safe_embedding(
            type_seqs, valid_event_mask, beta.t()
        )  # [B, L, K]

        # Broadcast for query × source × K × S
        alpha_src = alpha_src.unsqueeze(1).unsqueeze(3)  # [B, 1, L, 1, K]
        beta_src = beta_src.unsqueeze(1).unsqueeze(3)  # [B, 1, L, 1, K]
        tau = tau.unsqueeze(-1)  # [B, L_q, L_s, S, 1]

        excitation = (alpha_src * beta_src) * torch.exp(-beta_src * tau)

        # Causal mask — strictly lower triangular (past events only)
        if compute_last_step_only:
            causal = torch.ones(1, 1, L, 1, 1, device=dev)
        else:
            causal = torch.tril(torch.ones(L, L, device=dev), diagonal=-1).view(
                1, L, L, 1, 1
            )

        past_influence = (excitation * causal).sum(dim=2)  # [B, L_q, S, K]
        lambda_t = mu.view(1, 1, 1, -1) + past_influence

        return lambda_t

    # ──────────────────────────────────────────────────────────────────────────
    # Analytical integral of the intensity
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_integral_analytical(
        self,
        time_seq: torch.Tensor,  # [B, L]
        time_delta_seq: torch.Tensor,  # [B, L]
        type_seq: torch.Tensor,  # [B, L]  (already safe)
    ) -> torch.Tensor:
        B, L = time_seq.shape

        mu = self.mu_pos
        alpha = self.alpha_pos
        beta = self.beta_pos

        dt = time_delta_seq[:, 1:]  # [B, L-1]

        # Baseline contribution — Σ_k μ_k · dt_i
        integral_base = mu.sum() * dt  # [B, L-1]

        # Excitation contribution
        # tau_start[b, i, j] = time elapsed from event j to the start of interval i
        # We use cumulative deltas: tau_start[b, i, j] = Σ_{n=j+1}^{i} dt_n
        alpha_src = F.embedding(type_seq.long(), alpha.t()).unsqueeze(1)  # [B, 1, L, K]
        beta_src = F.embedding(type_seq.long(), beta.t()).unsqueeze(1)  # [B, 1, L, K]

        # Rebuild elapsed time from time_delta_seq for the integral bounds
        tau_cumsum = torch.cumsum(time_delta_seq, dim=1)  # [B, L]
        tau_start_full = (
            (
                tau_cumsum[:, 1:].unsqueeze(2) - tau_cumsum.unsqueeze(1)  # [B, L-1, L]
            )
            .unsqueeze(-1)
            .abs()
        )  # [B, L-1, L, 1]

        dt_4d = dt.unsqueeze(-1).unsqueeze(-1)  # [B, L-1, 1, 1]
        time_factor = 1.0 - torch.exp(-beta_src * dt_4d)  # [B, L-1, L, K]
        decay_start = torch.exp(-beta_src * tau_start_full)  # [B, L-1, L, K]

        term = alpha_src * time_factor * decay_start  # [B, L-1, L, K]

        # Causal mask — event j contributes to interval i only if j ≤ i
        causal = torch.tril(
            torch.ones(L - 1, L, device=time_seq.device), diagonal=0
        ).view(1, L - 1, L, 1)

        excitation_integral = (term * causal).sum(dim=-1).sum(dim=-1)  # [B, L-1]

        return integral_base + excitation_integral

    # ──────────────────────────────────────────────────────────────────────────
    # NLL loss
    # ──────────────────────────────────────────────────────────────────────────

    def loglike_loss(self, batch: Batch) -> tuple[torch.Tensor, int]:
        time_seq = batch.time_seqs
        type_seq = batch.type_seqs
        time_delta_seq = batch.time_delta_seqs
        mask = batch.valid_event_mask

        safe_types = type_seq.long().clone()
        safe_types[~mask.bool()] = 0

        # Full intensity over sequence — shape [B, L, 1, K] → squeeze → [B, L, K]
        intensities_full = self.compute_intensities_at_sample_dtimes(
            time_seqs=time_seq,
            type_seqs=safe_types,
            valid_event_mask=mask,
            compute_last_step_only=False,
        ).squeeze(-2)  # [B, L, K]

        # Target: events 1..L-1 (event 0 has no history)
        intensities_target = intensities_full[:, 1:, :]  # [B, L-1, K]
        target_types = safe_types[:, 1:].unsqueeze(-1)  # [B, L-1, 1]

        lambda_target = torch.gather(intensities_target, -1, target_types).squeeze(-1)
        event_ll = torch.log(lambda_target + 1e-9)

        integral = self._compute_integral_analytical(
            time_seq, time_delta_seq, safe_types
        )

        pad_mask = mask[:, 1:]
        event_ll = (event_ll * pad_mask).sum()
        non_event_ll = (integral * pad_mask).sum()
        num_events = int(pad_mask.sum().item())

        return -(event_ll - non_event_ll), num_events

    # ──────────────────────────────────────────────────────────────────────────
    # State synchronization
    # ──────────────────────────────────────────────────────────────────────────

    def sync_state(
        self,
        batch: Batch,
        t_current: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Compute the recursive excitation matrix R at time t_current.
        R[b, k, m] = Σ_{t_i < t_current, type_i = m} exp(-β[k, m] · (t_current - t_i))
        """
        dev = t_current.device
        mask = batch.valid_event_mask
        B, L = batch.time_seqs.shape
        K = self.num_event_types

        R = torch.zeros(B, K, K, dtype=torch.float32, device=dev)

        if not mask.any():
            return R

        safe_types = batch.type_seqs.long().clone()
        safe_types[~mask] = 0

        # [B, L] — elapsed time from each past event to t_current
        tau = (t_current.unsqueeze(1) - batch.time_seqs).clamp(min=0.0)

        # Decay:  exp(-β[k,m] · τ)  for each (event, k, m)  →  [B, L, K, K]
        decay = torch.exp(-self.beta_pos.view(1, 1, K, K) * tau.view(B, L, 1, 1))

        # One-hot source mask: which column m does each event activate?  → [B, L, 1, K]
        src_mask = F.one_hot(safe_types, K).float().view(B, L, 1, K)
        valid = mask.float().view(B, L, 1, 1)

        # R[b, k, m] = Σ_i decay[b,i,k,m] · src_mask[b,i,m] · valid[b,i]
        R = (decay * src_mask * valid).sum(dim=1)  # [B, K, K]

        return R

    # ──────────────────────────────────────────────────────────────────────────
    # Simulation — Ogata thinning, fully vectorised, GPU-compatible
    # ──────────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        batch: Batch,
        max_events: int = 10,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> SimulationResult:
        """
        Simulate via Ogata's thinning algorithm.

        Design principles
        -----------------
        - All tensor ops stay on `device` (GPU-compatible).
        - mu/alpha/beta softplus computed ONCE before the loop.
        - Variable-length storage uses Python lists + a single pad at the end.
        - No F.pad, no tqdm sync, no redundant .item() inside the hot loop.
        """
        dev = getattr(self, "device", torch.device("cpu"))
        K = self.num_event_types
        B = batch.time_seqs.size(0)

        with torch.no_grad():
            # ── Precompute positive params (outside the loop) ──
            mu = self.mu_pos  # [K]
            alpha = self.alpha_pos  # [K, K]
            beta = self.beta_pos  # [K, K]
            alpha_beta = alpha * beta  # [K, K]

            # ── Simulation window ──
            if start_time is None or end_time is None:
                st, et = self._compute_start_end_time(
                    batch.time_seqs, batch.valid_event_mask
                )
            start_times = (
                torch.full((B,), start_time, device=dev)
                if start_time is not None
                else st
            )
            end_times = (
                torch.full((B,), end_time, device=dev) if end_time is not None else et
            )

            # ── Initial excitation state ──
            R = self.sync_state(batch, t_current=start_times)  # [B, K, K]
            current_time = start_times.clone()
            last_event_t = start_times.clone()
            active = torch.ones(B, dtype=torch.bool, device=dev)

            # Variable-length event storage (CPU lists — avoids dynamic realloc on GPU)
            times_list: list[list[float]] = [[] for _ in range(B)]
            deltas_list: list[list[float]] = [[] for _ in range(B)]
            types_list: list[list[int]] = [[] for _ in range(B)]

            # ── Ogata thinning loop ──
            while active.any():
                # A. Upper-bound intensity  M[b] = Σ_k λ_k^{UB}(t)
                M = (
                    (mu + (alpha_beta * R).sum(dim=2)).sum(dim=1).clamp(min=self.eps)
                )  # [B]

                # B. Sample candidate arrival
                dt_prop = torch.empty(B, device=dev).exponential_(1.0) / M  # [B]
                t_cand = current_time + dt_prop

                active &= t_cand < end_times
                if not active.any():
                    break

                # C. Decay R to candidate time (all paths, active or not — no branching)
                R = R * torch.exp(-beta * dt_prop.view(B, 1, 1))

                # D. True intensity at t_cand
                lambda_k = mu + (alpha_beta * R).sum(dim=2)  # [B, K]
                lambda_sum = lambda_k.sum(dim=1)  # [B]

                # E. Thinning acceptance
                U = torch.rand(B, device=dev)
                accept = active & (U * M <= lambda_sum)
                current_time = t_cand

                if not accept.any():
                    continue

                # F. Sample event type for accepted paths
                acc_idx = accept.nonzero(as_tuple=True)[0]  # [n_acc]
                probs = lambda_k[acc_idx] / lambda_sum[acc_idx, None]  # [n_acc, K]
                k = torch.multinomial(probs, num_samples=1).squeeze(1)  # [n_acc]

                # G. Update excitation state R[b, :, k] += 1 for accepted paths
                R[acc_idx] += (
                    F.one_hot(k, K).float().unsqueeze(1)
                )  # broadcast over K rows

                # H. Record (transfer to CPU once per accepted batch, not per event)
                t_acc = t_cand[acc_idx].cpu().tolist()
                lt_acc = last_event_t[acc_idx].cpu().tolist()
                k_cpu = k.cpu().tolist()
                b_list = acc_idx.cpu().tolist()

                for n, b in enumerate(b_list):
                    if len(times_list[b]) >= max_events:
                        active[b] = False
                        continue
                    times_list[b].append(t_acc[n])
                    deltas_list[b].append(t_acc[n] - lt_acc[n])
                    types_list[b].append(k_cpu[n])

                last_event_t[acc_idx] = t_cand[acc_idx]

        # ── Pack variable-length lists into padded tensors ──
        max_len = max((len(ts) for ts in times_list), default=1)

        def _pad(lists: list[list], dtype: torch.dtype) -> torch.Tensor:
            out = torch.zeros(B, max_len, dtype=dtype, device=dev)
            for b, lst in enumerate(lists):
                if lst:
                    out[b, : len(lst)] = torch.tensor(lst, dtype=dtype, device=dev)
            return out

        lens = torch.tensor([len(ts) for ts in times_list], device=dev)
        valid_event_mask = torch.arange(max_len, device=dev)[None] < lens[:, None]

        return SimulationResult(
            time_seqs=_pad(times_list, torch.float32),
            time_delta_seqs=_pad(deltas_list, torch.float32),
            type_seqs=_pad(types_list, torch.long),
            valid_event_mask=valid_event_mask,
        )
