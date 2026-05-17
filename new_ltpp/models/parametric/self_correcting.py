"""
PyTorch implementation of the multivariate Self-Correcting (Epidemic) process model.

Intensity :
    λ_k(t) = exp(μ_k * t - Σ_{t_i < t} α[k, m_i])
"""

from typing import Optional

from new_ltpp.models.base import TrainingMixin
from new_ltpp.shared_types import Batch, SimulationResult

import torch
import torch.nn.functional as F
from torch import nn


class SelfCorrecting(TrainingMixin):
    """
    Multivariate Self-Correcting process with matrix-valued α parameters.

    Parameters are learnable (nn.Parameter), making this class compatible with
    gradient-based fitting via exact NLL minimisation.

    Args:
        mu   : Baseline growth rates.           Shape (K,)  or list of length K.
               μ_k > 0 (enforced with clamp to ensure valid integrals).
        alpha: Correction magnitudes.           Shape (K, K) or list of lists.
               α[k, m] = decrease in type-k log-intensity after a type-m event.
    """

    mu: nn.Parameter
    alpha: nn.Parameter

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        mu: Optional[list[float] | torch.Tensor] = None,
        alpha: Optional[list[list[float]] | torch.Tensor] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if mu is None:
            mu = torch.zeros(self.num_event_types)
        if alpha is None:
            alpha = torch.zeros((self.num_event_types, self.num_event_types))

        # On infère K (nombre de dimensions) à partir de la matrice
        if hasattr(self, "num_event_types"):
            K = self.num_event_types
        else:
            K = len(mu) if isinstance(mu, list) else mu.shape[0]
            self.num_event_types = K

        self.eps = 1e-5  # Protection contre la division par zéro dans l'intégrale

        def _to_param(x, shape, name):
            dev = getattr(self, "device", torch.device("cpu"))
            t = torch.tensor(x, dtype=torch.float32, device=dev).view(shape)
            if t.shape != torch.Size(shape):
                raise ValueError(
                    f"SelfCorrecting: expected {name} of shape {shape}, got {list(t.shape)}"
                )
            return t

        mu_t = _to_param(mu, (K,), "mu")
        alpha_t = _to_param(alpha, (K, K), "alpha")

        self.mu = nn.Parameter(mu_t)
        self.alpha = nn.Parameter(alpha_t)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_history(
        self,
        type_seqs: torch.Tensor,  # [B, L]
        valid_mask: torch.Tensor,  # [B, L] bool
    ) -> torch.Tensor:
        """
        Calcule la somme cumulée des corrections H(t) = Σ_{t_i < t} α[:, m_i].
        Retourne un tenseur H_past de shape [B, L, K] représentant la pénalité
        pour chaque type d'événement STRICTEMENT avant l'événement j.
        """
        B, L = type_seqs.shape
        K = self.num_event_types

        safe_types = type_seqs.long().clone()
        safe_types[~valid_mask.bool()] = 0

        # [B, L, K]
        alpha_emb = F.embedding(safe_types, self.alpha.t())
        alpha_emb = alpha_emb * valid_mask.float().unsqueeze(-1)

        # Somme cumulée sur la séquence temporelle
        H = torch.cumsum(alpha_emb, dim=1)

        # Décalage pour obtenir l'historique strict (avant t_j)
        zeros = torch.zeros(B, 1, K, device=H.device, dtype=H.dtype)
        H_past = torch.cat([zeros, H[:, :-1, :]], dim=1)

        return H_past

    # ──────────────────────────────────────────────────────────────────────────
    # Core intensity computation
    # ──────────────────────────────────────────────────────────────────────────

    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Vectorised multivariate Self-Correcting intensity.

        λ_k(t) = exp(μ_k * t - Σ_{t_i < t} α[k, m_i])
        """
        H_past = self._get_history(type_seqs, valid_event_mask)  # [B, L, K]

        if compute_last_step_only:
            H_past = H_past[:, -1:, :]
            time_seqs = time_seqs[:, -1:]

        # Absolute time: [B, L, S, 1]
        T = (time_seqs.unsqueeze(-1) + sample_dtimes).unsqueeze(-1)
        H = H_past.unsqueeze(2)  # [B, L, 1, K]

        mu = torch.clamp(self.mu, min=self.eps)  # [K]

        # État du compensateur: x(t) = μ*t - H
        x = T * mu - H

        return torch.exp(x)

    # ──────────────────────────────────────────────────────────────────────────
    # Log-likelihood loss
    # ──────────────────────────────────────────────────────────────────────────

    def loglike_loss(self, batch: Batch) -> tuple[torch.Tensor, int]:
        """
        Exact NLL for the multivariate Self-Correcting process.

        LL = Σ_i log λ_{k_i}(t_i)  −  Σ_j ∫_{t_j}^{t_{j+1}} Σ_k λ_k(t) dt
        """
        time_seq = batch.time_seqs  # [B, L]
        type_seq = batch.type_seqs  # [B, L]
        time_delta_seq = batch.time_delta_seqs  # [B, L]
        mask = batch.valid_event_mask  # [B, L] bool

        safe_types = type_seq.long().clone()
        safe_types[~mask.bool()] = 0

        # Historique strict pour toute la séquence
        H_past = self._get_history(type_seq, mask)  # [B, L, K]
        mu = torch.clamp(self.mu, min=self.eps)  # [K]

        # ── 1. Log-intensity at each observed event ───────────────────────────
        # Événements 1..N
        T_obs = time_seq[:, 1:]  # [B, L-1]
        H_obs = H_past[:, 1:, :]  # [B, L-1, K]

        x_obs = T_obs.unsqueeze(-1) * mu - H_obs  # [B, L-1, K]

        target_types = safe_types[:, 1:].unsqueeze(-1)  # [B, L-1, 1]
        x_target = torch.gather(x_obs, dim=-1, index=target_types).squeeze(-1)

        # Puisque λ = exp(x), log(λ) = x, ce qui est numériquement parfait
        event_ll = x_target  # [B, L-1]

        # ── 2. Analytical integral over each inter-event interval ─────────────
        # Intervales de t_0..t_{L-2}
        T_start = time_seq[:, :-1]  # [B, L-1]
        H_start = H_past[:, :-1, :]  # [B, L-1, K]
        dt = time_delta_seq[:, 1:].unsqueeze(-1)  # [B, L-1, 1]

        x_start = T_start.unsqueeze(-1) * mu - H_start  # [B, L-1, K]

        # ∫ exp(x_start + μ*s) ds = exp(x_start) * (exp(μ*Δt) - 1) / μ
        integral_k = torch.exp(x_start) * (torch.exp(mu * dt) - 1.0) / mu
        integral = integral_k.sum(dim=-1)  # [B, L-1]

        # ── 3. Masked reduction ───────────────────────────────────────────────
        pad_mask = mask[:, 1:]  # [B, L-1]

        event_ll = (event_ll * pad_mask).sum()
        non_event_ll = (integral * pad_mask).sum()

        num_events = int(pad_mask.sum().item())
        loss = -(event_ll - non_event_ll)

        return loss, num_events

    # ──────────────────────────────────────────────────────────────────────────
    # State Synchronization & Simulation
    # ──────────────────────────────────────────────────────────────────────────

    def sync_state(
        self, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synchronise l'état interne (H, t_current, t_end) à partir d'un batch d'historique.
        Reproduit la logique de fenêtre de simulation du Simulator générique.

        Returns:
            H: Tenseur [B, K] de la somme des pénalités alpha pour les événements passés.
            t_current: Tenseur [B] du temps actuel (end_times du batch).
            t_end: Tenseur [B] de la fin de simulation cible.
        """
        mask = batch.valid_event_mask
        time_seqs = batch.time_seqs
        type_seqs = batch.type_seqs

        # ── 1. Calcul de H (Somme des pénalités passées) ──
        safe_types = type_seqs.long().clone()
        safe_types[~mask] = 0

        # [B, L, K]
        alpha_emb = F.embedding(safe_types, self.alpha.t())
        alpha_emb = alpha_emb * mask.float().unsqueeze(-1)

        # On somme sur l'axe temporel (L) -> [B, K]
        H = alpha_emb.sum(dim=1)

        # ── 2. Calcul des temps (start, current, horizon) ──
        time_clone_min = time_seqs.clone()
        time_clone_min[~mask] = float("inf")
        start_times = time_clone_min.min(dim=1).values
        start_times[torch.isinf(start_times)] = 0.0  # Fallback si la séquence est vide

        time_clone_max = time_seqs.clone()
        time_clone_max[~mask] = 0.0
        end_times = time_clone_max.max(dim=1).values

        # La durée à simuler est égale à la durée de l'historique fourni
        sim_window = end_times - start_times

        # Protection si le batch est complètement vide (simulate_from_scratch)
        # On garantit une fenêtre minimale strictement positive (ex: 10.0)
        sim_window = torch.clamp(sim_window, min=1e-3)

        t_end = end_times + sim_window

        return H, end_times, t_end

    def simulate(
        self,
        batch: Batch,
        max_events: int = 10_000,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Batch:
        """
        Simulate the multivariate Self-Correcting process conditionné sur un batch.
        Utilise l'inversion exacte vectorisée sur tout le batch.
        """
        dev = getattr(self, "device", torch.device("cpu"))
        K = self.num_event_types
        batch_size = batch.time_seqs.size(0)

        mu = torch.clamp(self.mu.detach(), min=self.eps)  # [K]
        alpha_t = self.alpha.detach().t()  # [K_source, K_target]

        # ── 1. Synchronisation de l'état avec le batch conditionnel ──
        H, t, t_end = self.sync_state(batch)
        H = H.detach()
        t = t.detach()
        t_end = t_end.detach()

        active = torch.ones(batch_size, dtype=torch.bool, device=dev)  # [B]

        # Pré-allocation
        all_times = torch.zeros(
            (batch_size, max_events), dtype=torch.float32, device=dev
        )
        all_deltas = torch.zeros(
            (batch_size, max_events), dtype=torch.float32, device=dev
        )
        all_types = torch.zeros((batch_size, max_events), dtype=torch.long, device=dev)
        lens = torch.zeros(batch_size, dtype=torch.long, device=dev)

        # ── 2. Boucle de simulation vectorisée ──
        while active.any():
            x = mu * t.unsqueeze(1) - H  # [B, K]

            E = torch.empty((batch_size, K), device=dev).exponential_(1.0)
            tau = torch.log(E * mu / torch.exp(x) + 1.0) / mu  # [B, K]

            dt, next_dim = torch.min(tau, dim=1)  # dt: [B], next_dim: [B]
            t_next = t + dt

            # Vectorized condition: t_next doit être inférieur au t_end de SA séquence
            valid_step = active & (t_next < t_end) & (lens < max_events)
            active = valid_step

            if not active.any():
                break

            t = torch.where(valid_step, t_next, t)
            H[valid_step] = H[valid_step] + alpha_t[next_dim[valid_step]]

            b_idx = torch.arange(batch_size, device=dev)[valid_step]
            curr_lens = lens[valid_step]

            all_times[b_idx, curr_lens] = t_next[valid_step]
            all_deltas[b_idx, curr_lens] = dt[valid_step]
            all_types[b_idx, curr_lens] = next_dim[valid_step]

            lens[valid_step] += 1

        # ── 3. Post-traitement et renvoi du Batch généré ──
        max_len = max(lens.max().item(), 1)

        time_seqs = all_times[:, :max_len]
        time_delta_seqs = all_deltas[:, :max_len]
        type_seqs = all_types[:, :max_len]

        seq_idx = torch.arange(max_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        valid_event_mask = seq_idx < lens.unsqueeze(1)

        return SimulationResult(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            valid_event_mask=valid_event_mask,
        )
