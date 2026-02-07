from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from new_ltpp.models.base_model import Model
from new_ltpp.shared_types import Batch
from new_ltpp.shared_types import SimulationResult


class SelfCorrecting(Model):
    """
    PyTorch implementation of the Self-Correcting Point Process model.
    Intensity for type i: lambda_i(t) = exp(mu_i + alpha_i * (t - N_i(t)))
    where N_i(t) is the number of events of type i occurred strictly before time t.
    """

    def __init__(
        self,
        *,
        mu: list[float] | torch.Tensor,
        alpha: list[float] | torch.Tensor,
        **kwargs,
    ):
        super(SelfCorrecting, self).__init__(**kwargs)

        # Validate and convert parameters
        mu = torch.as_tensor(mu, dtype=torch.float32, device=self.device)
        alpha = torch.as_tensor(alpha, dtype=torch.float32, device=self.device)

        if (
            mu.shape[0] != self.num_event_types
            or alpha.shape[0] != self.num_event_types
        ):
            raise ValueError(
                f"Dimension mismatch. Expected ({self.num_event_types},). "
                f"Got mu: {mu.shape}, alpha: {alpha.shape}"
            )

        self.mu = nn.Parameter(mu.view(self.num_event_types))
        self.alpha = nn.Parameter(alpha.view(self.num_event_types))

    def _get_cumulative_counts(self, type_seq: torch.Tensor) -> torch.Tensor:
        """
        Efficiently compute N_i(t) for each time step.

        Returns:
            torch.Tensor: [Batch, Seq_Len, Num_Types]
            counts[b, k, i] = number of events of type i in type_seq[b, :k+1]
        """
        # [Batch, Seq, Num_Types]
        type_one_hot = F.one_hot(
            type_seq.long(), num_classes=self.num_event_types
        ).float()

        # Cumsum along the time dimension
        cumulative_counts = torch.cumsum(type_one_hot, dim=1)
        return cumulative_counts

    def compute_intensities_at_sample_times(
        self,
        *,
        time_seq: torch.Tensor,
        type_seq: torch.Tensor,
        sample_dtimes: torch.Tensor,
        valid_event_mask: Optional[
            torch.Tensor
        ] = None,  # Not used in SelfCorrecting but kept for compatibility
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute lambda(t + delta) in a vectorized way.
        """
        # 1. Retrieve historical counts N_i(t)
        # counts_history[b, k] contains the count INCLUDING event k.
        counts_at_events = self._get_cumulative_counts(type_seq)  # [B, L, D]

        if compute_last_step_only:
            # Take the time of the last event and the associated counts
            base_time = time_seq[:, -1:].unsqueeze(-1)  # [B, 1, 1]
            base_counts = counts_at_events[:, -1:, :].unsqueeze(2)  # [B, 1, 1, D]

            # If sample_dtimes is [B, 1, N_samples] or [B, L, N_samples], adapt accordingly
            if sample_dtimes.dim() == 3 and sample_dtimes.shape[1] != 1:
                sample_dtimes = sample_dtimes[:, -1:, :]

            # Absolute time t = t_last + delta
            # [B, 1, N_samples] -> [B, 1, N_samples, 1]
            current_times = (base_time + sample_dtimes).unsqueeze(-1)

        else:
            # For the whole sequence
            base_time = time_seq.unsqueeze(-1)  # [B, L, 1]
            base_counts = counts_at_events.unsqueeze(2)  # [B, L, 1, D]

            # Absolute time t = t_k + delta
            # [B, L, N_samples, 1]
            current_times = (base_time + sample_dtimes).unsqueeze(-1)

        # 2. Compute intensity
        # Formula: exp(mu + alpha * (t - N(t)))
        # N(t) here is the number of events *strictly before* t.
        # Since t > t_k (because delta > 0), N(t) includes event k.
        # Therefore base_counts (which is the cumsum including k) is correct.

        # [1, 1, 1, D]
        mu = self.mu.view(1, 1, 1, -1)
        alpha = self.alpha.view(1, 1, 1, -1)

        # exponent: [B, L, N_samples, D]
        # Broadcasting: t (..., 1) - N (..., D)
        exponent = mu + alpha * (current_times - base_counts)

        intensities = torch.exp(exponent)

        # Optional numerical safety
        # intensities = torch.clamp(intensities, min=1e-9)

        return intensities

    def loglike_loss(self, batch: Batch) -> Tuple[torch.Tensor, int]:
        """
        Compute the exact (analytical) log-likelihood.
        LL = sum(log(lambda(t_i))) - int(lambda(t) dt)
        """
        time_seq = batch.time_seqs
        type_seq = batch.type_seqs

        # Mask (ignore padding and the first event which is just an anchor t0)
        # [Batch, L-1]
        seq_mask = batch.valid_event_mask[:, 1:]

        # --- Data Preparation ---

        # 1. Counters N(t)
        # [Batch, L, D]
        all_counts = self._get_cumulative_counts(type_seq)

        # To predict event k (at time t_k), use history up to k-1.
        # Therefore N(t_k) = counts[k-1]
        # [Batch, L-1, D]
        N_prev = all_counts[:, :-1, :]

        # Target event times t_1 ... t_N
        # [Batch, L-1, 1]
        t_target = time_seq[:, 1:].unsqueeze(-1)

        # Paramètres reshaped [1, 1, D]
        mu = self.mu.view(1, 1, -1)
        alpha = self.alpha.view(1, 1, -1)

        # --- A. Event Log-Likelihood ---

        # lambda(t_k) = exp(mu + alpha * (t_k - N(t_k)))
        # [Batch, L-1, D]
        exponent_at_event = mu + alpha * (t_target - N_prev)
        lambda_at_event = torch.exp(exponent_at_event)

        # Select the intensity of the type that actually occurred
        target_types = type_seq[:, 1:].long().unsqueeze(-1)  # [Batch, L-1, 1]

        # [Batch, L-1]
        lambda_target = torch.gather(
            lambda_at_event, dim=-1, index=target_types
        ).squeeze(-1)

        # Log(lambda)
        event_ll = torch.log(lambda_target + 1e-9)

        # --- B. Integral (Non-Event Log-Likelihood) ---

        # Integrate over intervals [t_{k-1}, t_k].
        # Interval duration: t_k - t_{k-1} (we use absolute times in the formula)
        # In the interval (t_{k-1}, t_k), the count N(t) is constant and equals N_prev (counts up to k-1).

        t_start = time_seq[:, :-1].unsqueeze(-1)  # t_{k-1}

        # Analytical calculation of the integral over [t_start, t_end]
        # Int = (1/alpha) * [ lambda(t_end) - lambda(t_start) ]
        # Note: lambda here is computed with the current N (N_prev)

        # Lambda at the start of the interval (just after the previous event)
        lambda_start = torch.exp(mu + alpha * (t_start - N_prev))

        # Lambda at the end of the interval (just before the current event)
        # This is exactly `lambda_at_event` computed above.
        lambda_end = lambda_at_event

        # Integral per type: [Batch, L-1, D]
        # Handle division by zero if alpha ~ 0 (optional, here we assume alpha != 0)
        # For stability: alpha + epsilon
        alpha_safe = alpha + 1e-9 * torch.sign(alpha)

        integral_per_type = (lambda_end - lambda_start) / alpha_safe

        # Sum over all types D
        # [Batch, L-1]
        non_event_ll = integral_per_type.sum(dim=-1)

        # --- C. Total Loss ---

        # Masked sum
        loss_event = (event_ll * seq_mask).sum()
        loss_non_event = (non_event_ll * seq_mask).sum()

        num_events = seq_mask.sum().item()

        # NLL
        loss = -(loss_event - loss_non_event)

        return loss, int(num_events)

    def simulate(
        self,
        batch: Optional[Batch] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        batch_size: Optional[int] = None,
        initial_buffer_size: Optional[int] = None,
    ) -> SimulationResult:
        """
        Simulate sequences of event types.
        """
        raise NotImplementedError(
            "Simulation not implemented for Self-Correcting model, there is a custom implementation in the data generation class. This class serves as a benchmark for the other models in prediction phase for the loglike loss."
        )
