"""
PyTorch implementation of the multivariate Hawkes process model.

Used as a parametric benchmark against neural TPP models — supports exact
log-likelihood computation and simulation via Ogata's thinning algorithm.

Intensity:
    λ_k(t) = μ_k + Σ_{t_i < t} α[k, m_i] · β[k, m_i] · exp(−β[k, m_i] · (t − t_i))
    (where μ_k, α, β are constrained to be positive via squaring)
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
            is_user_provided = x is not None
            if not is_user_provided:
                x = default
            t = torch.as_tensor(x, dtype=torch.float32, device=dev).view(shape)
            if t.shape != torch.Size(shape):
                raise ValueError(
                    f"Hawkes: expected {name} of shape {shape}, got {list(t.shape)}"
                )
            # Apply inverse square so the effective parameter is exactly what was requested (or default)
            t = torch.clamp(t, min=1e-7)
            t = torch.sqrt(t)
            return t

        self.mu = nn.Parameter(_to_param(mu, torch.full((K,), 0.01), (K,), "mu"))
        self.alpha = nn.Parameter(
            _to_param(alpha, torch.full((K, K), 0.01), (K, K), "alpha")
        )
        self.beta = nn.Parameter(_to_param(beta, torch.ones((K, K)), (K, K), "beta"))

    # ──────────────────────────────────────────────────────────────────────────
    # Properties — positive-constrained parameters
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def mu_pos(self) -> torch.Tensor:
        return torch.square(self.mu)

    @property
    def alpha_pos(self) -> torch.Tensor:
        return torch.square(self.alpha)

    @property
    def beta_pos(self) -> torch.Tensor:
        return torch.square(self.beta)

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

    def _compute_total_integral(
        self,
        time_seq: torch.Tensor,  # [B, L]
        type_seq: torch.Tensor,  # [B, L]
        mask: torch.Tensor,      # [B, L]
    ) -> torch.Tensor:
        """Computes the total integral of the intensity over the observation window [0, T]."""
        B, L = time_seq.shape
        mu = self.mu_pos
        alpha = self.alpha_pos
        beta = self.beta_pos

        # Observation window ends at the last valid event time for each sequence
        lengths = mask.sum(dim=1).long()  # [B]
        last_indices = (lengths - 1).clamp(min=0)
        T = time_seq[torch.arange(B, device=time_seq.device), last_indices]  # [B]

        # Baseline integral: Σ_k μ_k * T
        integral_base = mu.sum() * T  # [B]

        # Excitation integral
        tau = T.unsqueeze(1) - time_seq  # [B, L]
        tau = tau.clamp(min=0.0)

        alpha_src = F.embedding(type_seq.long(), alpha.t())  # [B, L, K]
        beta_src = F.embedding(type_seq.long(), beta.t())  # [B, L, K]

        # term = α * (1 - exp(-β * (T - t_i)))
        tau_3d = tau.unsqueeze(-1)  # [B, L, 1]
        excitation = alpha_src * (1.0 - torch.exp(-beta_src * tau_3d))  # [B, L, K]
        
        # Only valid events contribute to the excitation integral
        excitation = excitation * mask.float().unsqueeze(-1)  # [B, L, K]

        integral_excitation = excitation.sum(dim=[1, 2])  # [B]

        return integral_base + integral_excitation

    # ──────────────────────────────────────────────────────────────────────────
    # NLL loss
    # ──────────────────────────────────────────────────────────────────────────

    def loglike_loss(self, batch: Batch) -> tuple[torch.Tensor, int]:
        time_seq = batch.time_seqs
        type_seq = batch.type_seqs
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

        # Target ALL valid events
        target_types = safe_types.unsqueeze(-1)  # [B, L, 1]
        lambda_target = torch.gather(intensities_full, -1, target_types).squeeze(-1) # [B, L]
        
        event_ll = torch.log(lambda_target + 1e-9)
        event_ll = (event_ll * mask).sum()

        integral = self._compute_total_integral(time_seq, safe_types, mask)
        non_event_ll = integral.sum()

        num_events = int(mask.sum().item())

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
        num_events_to_simulate: Optional[int] = None,
    ) -> SimulationResult:
        """
        Simulate via Ogata's thinning algorithm.

        Design principles
        -----------------
        - All tensor ops stay on `device` (GPU-compatible).
        - mu/alpha/beta softplus computed ONCE before the loop.
        - Exactly num_events_to_simulate events are generated for each sequence.
        """
        dev = getattr(self, "device", torch.device("cpu"))
        K = self.num_event_types
        B = batch.time_seqs.size(0)
        
        if num_events_to_simulate is None:
            num_events_to_simulate = batch.time_seqs.size(1)
            if num_events_to_simulate == 0:
                num_events_to_simulate = 100

        with torch.no_grad():
            # ── Precompute positive params (outside the loop) ──
            mu = self.mu_pos  # [K]
            alpha = self.alpha_pos  # [K, K]
            beta = self.beta_pos  # [K, K]
            alpha_beta = alpha * beta  # [K, K]

            # ── Simulation window ──
            if batch.time_seqs.size(1) == 0:
                start_times = torch.zeros(B, dtype=torch.float32, device=dev)
            else:
                time_clone = batch.time_seqs.clone()
                time_clone[~batch.valid_event_mask] = 0.0
                start_times = time_clone.max(dim=1).values

            # ── Initial excitation state ──
            R = self.sync_state(batch, t_current=start_times)  # [B, K, K]
            current_time = start_times.clone()
            last_event_t = start_times.clone()
            active = torch.ones(B, dtype=torch.bool, device=dev)

            # ── Pre-allocate tensors ──
            all_times = torch.zeros((B, num_events_to_simulate), dtype=torch.float32, device=dev)
            all_deltas = torch.zeros((B, num_events_to_simulate), dtype=torch.float32, device=dev)
            all_types = torch.zeros((B, num_events_to_simulate), dtype=torch.long, device=dev)
            lens = torch.zeros(B, dtype=torch.long, device=dev)

            # ── Ogata thinning loop ──
            while active.any():
                # A. Upper-bound intensity  M[b] = Σ_k λ_k^{UB}(t)
                M = ((mu + (alpha_beta * R).sum(dim=2)).sum(dim=1).clamp(min=self.eps))  # [B]

                # B. Sample candidate arrival
                dt_prop = torch.empty(B, device=dev).exponential_(1.0) / M  # [B]
                t_cand = current_time + dt_prop

                # C. Decay R to candidate time
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
                R[acc_idx] += F.one_hot(k, K).float().unsqueeze(1)

                # H. Record accepted events
                curr_lens = lens[acc_idx]
                dt_acc = t_cand[acc_idx] - last_event_t[acc_idx]
                
                all_times[acc_idx, curr_lens] = t_cand[acc_idx]
                all_deltas[acc_idx, curr_lens] = dt_acc
                all_types[acc_idx, curr_lens] = k

                last_event_t[acc_idx] = t_cand[acc_idx]
                lens[acc_idx] += 1

                # Deactivate paths that have reached the required number of events
                active[acc_idx] = lens[acc_idx] < num_events_to_simulate

        if batch.time_seqs.size(1) > 0:
            all_times = all_times - start_times.unsqueeze(1)

        valid_event_mask = torch.ones_like(all_times, dtype=torch.bool)

        return SimulationResult(
            time_seqs=all_times,
            time_delta_seqs=all_deltas,
            type_seqs=all_types,
            valid_event_mask=valid_event_mask,
        )
