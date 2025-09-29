import math

import torch
import torch.nn.functional as F  # Added import
from torch import nn

from easy_tpp.configs import ModelConfig
from easy_tpp.models.basemodel import Model


class SelfCorrecting(Model):
    """
    PyTorch implementation of the Self-Correcting Point Process model.
    Intensity for type i: lambda_i(t) = exp(mu_i + alpha_i * (t - N_i(t)))
    where N_i(t) is the number of events of type i occurred strictly before time t.
    Inherits from Model.
    """

    def __init__(self, model_config: ModelConfig, num_event_types: int, **kwargs):
        """
        Initialize the Self-Correcting model.

        Args:
            model_config (EasyTPP.ModelConfig): Configuration object containing model specs.
                Expected specs:
                - 'mu' (list or tensor): Base log-intensity parameter for each type.
                - 'alpha' (list or tensor): Correction factor for each type (often negative).
        """

        self.num_event_types = num_event_types

        if "mu" not in model_config.specs or "alpha" not in model_config.specs:
            raise ValueError(
                "SelfCorrecting model requires 'mu' and 'alpha' in model_config.specs"
            )

        # Convert parameters to tensors and move to the correct device
        mu = torch.tensor(model_config.specs["mu"], dtype=torch.float32)
        alpha = torch.tensor(model_config.specs["alpha"], dtype=torch.float32)

        if (
            mu.shape[0] != self.num_event_types
            or alpha.shape[0] != self.num_event_types
        ):
            raise ValueError(
                f"SelfCorrecting parameter dimension mismatch. Expected mu/alpha: ({self.num_event_types},). "
                f"Got mu: {mu.shape}, alpha: {alpha.shape}"
            )

        # Register parameters as buffers (non-trainable)
        self.register_buffer("mu", mu)  # Shape [D]
        self.register_buffer("alpha", alpha)  # Shape [D]

    def _compute_N_t(
        self, time_seq: torch.Tensor, type_seq: torch.Tensor, query_times: torch.Tensor
    ):
        """
        Computes N_i(t) = count of events of type i with timestamp < t.

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L_hist].
            type_seq (torch.Tensor): Event types [B, L_hist]. Assumes padding uses self.pad_token_id.
            query_times (torch.Tensor): Times at which to compute counts. Shape [B, L_query].

        Returns:
            torch.Tensor: Counts N_i(t) for each type i. Shape [B, L_query, D].
        """
        batch_size, seq_len_hist = time_seq.shape
        _, seq_len_query = query_times.shape
        num_types = self.num_event_types
        device = self.device

        # Expand dimensions for broadcasting
        query_times_exp = query_times.unsqueeze(-1).to(device)  # [B, L_query, 1]
        time_seq_exp = time_seq.unsqueeze(1).to(device)  # [B, 1, L_hist]
        type_seq_exp = type_seq.unsqueeze(1).to(device)  # [B, 1, L_hist]

        # Create mask for events happening *before* query times
        # Shape: [B, L_query, L_hist]
        before_query_mask = (time_seq_exp < query_times_exp) & (
            type_seq_exp != self.pad_token_id
        )

        # Create one-hot encoding for event types in history
        # type_seq_exp: [B, 1, L_hist] -> one_hot -> [B, 1, L_hist, D] (D = num_event_types)
        # Note: Need to handle pad_token_id if it's outside the range [0, D-1]
        # Assuming types are 0 to D-1. If pad_token_id is large, clamp or handle separately.
        safe_type_seq = type_seq_exp.clone()
        pad_mask = safe_type_seq == self.pad_token_id
        safe_type_seq[pad_mask] = 0  # Temporarily set pad to 0 for one_hot
        type_one_hot = F.one_hot(
            safe_type_seq.long(), num_classes=self.num_event_types
        ).float()  # [B, 1, L_hist, D]
        type_one_hot[pad_mask.unsqueeze(-1).expand_as(type_one_hot)] = (
            0  # Zero out one-hot for padded events
        )

        # Combine masks: count event k if it's before query time t and has type i
        # before_query_mask: [B, L_query, L_hist, 1]
        # type_one_hot:      [B, 1,       L_hist, D]
        # Combined mask shape: [B, L_query, L_hist, D]
        count_mask = before_query_mask.unsqueeze(-1) * type_one_hot

        # Sum over history dimension (L_hist) to get counts N_i(t)
        # Shape: [B, L_query, D]
        N_t = count_mask.sum(dim=2)

        return N_t

    def compute_intensities_at_times(
        self,
        time_seq: torch.Tensor,
        time_delta_seq: torch.Tensor,  # Not directly used here
        type_seq: torch.Tensor,
        query_times: torch.Tensor,
        **kwargs,
    ):
        """
        Computes the intensity lambda_i(t) = exp(mu_i + alpha_i * (t - N_i(t)))
        for all event types at specified query times.

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L_hist].
            type_seq (torch.Tensor): Event types [B, L_hist].
            query_times (torch.Tensor): Times at which to compute intensities. Shape [B, L_query] or [B, L_query, N_samples].

        Returns:
            torch.Tensor: Intensities lambda_i(t) for each type i. Shape [B, L_query, D] or [B, L_query, N_samples, D].
        """
        batch_size, seq_len_hist = time_seq.shape
        device = self.device

        # Handle different query_times shapes
        if query_times.dim() == 2:
            # [B, L_query] -> [B, L_query, 1]
            query_times = query_times.unsqueeze(-1)
            has_samples = False
        else:
            # [B, L_query, N_samples]
            has_samples = True

        batch_size, seq_len_query, num_samples = query_times.shape
        num_types = self.num_event_types

        # Compute N_i(t) for all query times and types
        # Reshape query_times for _compute_N_t: [B, L_query * N_samples]
        query_times_flat = query_times.view(batch_size, -1)

        # N_t shape: [B, L_query * N_samples, D]
        N_t = self._compute_N_t(time_seq, type_seq, query_times_flat)

        # Reshape back to [B, L_query, N_samples, D]
        N_t = N_t.view(batch_size, seq_len_query, num_samples, num_types)

        # Get parameters mu and alpha, ensure they are on the correct device
        mu_dev = self.mu.to(device)  # [D]
        alpha_dev = self.alpha.to(device)  # [D]

        # Expand query_times for element-wise calculation: [B, L_query, N_samples, 1]
        query_times_exp = query_times.unsqueeze(-1).to(device)

        # Calculate the exponent term: mu_i + alpha_i * (t - N_i(t))
        # Shapes: mu_dev[None, None, None, :]: [1, 1, 1, D]
        #         alpha_dev[None, None, None, :]: [1, 1, 1, D]
        #         query_times_exp: [B, L_query, N_samples, 1]
        #         N_t: [B, L_query, N_samples, D]
        exponent = mu_dev + alpha_dev * (
            query_times_exp - N_t
        )  # Shape [B, L_query, N_samples, D]

        # Compute intensity: exp(exponent)
        intensities = torch.exp(exponent)

        # Ensure non-negative intensities
        intensities = torch.clamp(intensities, min=self.eps)

        # Remove N_samples dimension if it wasn't in the input
        if not has_samples:
            intensities = intensities.squeeze(2)

        return intensities

    def compute_intensities_at_sample_times(
        self,
        time_seq: torch.Tensor,
        time_delta_seq: torch.Tensor,  # Not directly used
        type_seq: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ):
        """
        Computes intensities at sampled times relative to each event in the sequence.
        Required by Model for prediction and loss calculation.
        Calculates lambda(t_k + delta_t) using history up to event k.

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L].
            time_delta_seq (torch.Tensor): Time differences [B, L].
            type_seq (torch.Tensor): Event types [B, L].
            sample_dtimes (torch.Tensor): Sampled time deltas relative to each event [B, L, N_samples].
            compute_last_step_only (bool): If True, only compute for the last event.

        Returns:
            torch.Tensor: Intensities lambda_i(t_k + delta_t). Shape [B, L, N_samples, D] or [B, 1, N_samples, D].
        """
        batch_size, seq_len = time_seq.shape
        num_samples = sample_dtimes.shape[-1]
        device = self.device

        time_seq = time_seq.to(device)
        type_seq = type_seq.to(device)
        sample_dtimes = sample_dtimes.to(device)

        if compute_last_step_only:
            # History is the full sequence
            hist_time_seq = time_seq
            hist_type_seq = type_seq
            # Query times are relative to the *last* event time
            # query_times shape: [B, 1, N_samples]
            query_times = time_seq[:, -1:].unsqueeze(-1) + sample_dtimes[:, -1:, :]

            # Compute intensities at these query times using the full history
            # Output shape: [B, 1, N_samples, D]
            intensities = self.compute_intensities_at_times(
                time_seq=hist_time_seq,
                time_delta_seq=None,
                type_seq=hist_type_seq,
                query_times=query_times,
            )

        else:
            # Compute for every step in the sequence
            # Query times are t_k + delta_t_n for each k and sample n
            # query_times shape: [B, L, N_samples]
            query_times = time_seq.unsqueeze(-1) + sample_dtimes

            # Use the vectorized approach similar to Hawkes
            intensities = self.compute_intensities_at_times(
                time_seq=time_seq,
                time_delta_seq=None,
                type_seq=type_seq,
                query_times=query_times,
            )

        return intensities

    def loglike_loss(self, batch: tuple) -> tuple[torch.Tensor, int]:
        """
        Compute the log-likelihood loss for the Self-Correcting process.
        Uses the same structure as the Hawkes model.

        Args:
            batch: Tuple containing time_seq, time_delta_seq, type_seq, batch_non_pad_mask, _

        Returns:
            tuple: (total negative log-likelihood loss, number of events)
        """
        time_seq_BN, time_delta_seq_BN, type_seq_BN, batch_non_pad_mask_BN, _ = batch

        # For lambda_at_event: intensity at actual event times t_1, ..., t_N
        # query_times_for_event_ll: [B, L-1, 1] where L is seq_len
        # These are absolute timestamps t_1, ..., t_N
        query_times_for_event_ll = time_seq_BN[:, 1:].unsqueeze(-1)

        # lambda_at_event: Intensities at events t_1, ..., t_N. Shape: [B, L-1, D]
        # History for these queries is events up to the previous step
        # We need to compute intensities at each event using history up to that point
        batch_size, seq_len = time_seq_BN.shape
        lambda_at_event = torch.zeros(
            batch_size, seq_len - 1, self.num_event_types, device=self.device
        )

        for k in range(1, seq_len):
            # History up to (but not including) event k
            hist_time_seq_k = time_seq_BN[:, :k]
            hist_type_seq_k = type_seq_BN[:, :k]

            # Query time is t_k
            query_time_k = time_seq_BN[:, k : k + 1].unsqueeze(-1)  # [B, 1, 1]

            # Compute intensities at t_k using history up to k-1
            intensities_k = self.compute_intensities_at_times(
                time_seq=hist_time_seq_k,
                time_delta_seq=None,
                type_seq=hist_type_seq_k,
                query_times=query_time_k,
            ).squeeze(
                -2
            )  # [B, 1, D] -> [B, D]

            lambda_at_event[:, k - 1, :] = intensities_k

        # For integral term: intervals (t_0,t_1), ..., (t_{N-1}, t_N)
        # time_delta_seq_BN[:, 1:] gives dt_1, ..., dt_N. Shape [B, L-1]
        time_delta_seq_for_integral = time_delta_seq_BN[:, 1:]

        # dts_samples_for_integral: Samples within each interval (t_0,t_1), ..., (t_{N-1}, t_N)
        # Shape: [B, L-1, G]
        dts_samples_for_integral = self.make_dtime_loss_samples(
            time_delta_seq_for_integral
        )

        # History for integral calculation: events up to t_{N-1}
        # For interval (t_k, t_{k+1}), use history up to t_k
        time_seq_hist_for_integral = time_seq_BN[:, :-1]
        type_seq_hist_for_integral = type_seq_BN[:, :-1]

        # lambdas_loss_samples: Intensities at sampled times within intervals. Shape: [B, L-1, G, D]
        lambdas_loss_samples = self.compute_intensities_at_sample_times(
            time_seq=time_seq_hist_for_integral,
            time_delta_seq=None,
            type_seq=type_seq_hist_for_integral,
            sample_dtimes=dts_samples_for_integral,
            compute_last_step_only=False,
        )

        # Prepare other arguments for compute_loglikelihood
        # These correspond to events t_1, ..., t_N
        type_seq_for_loss = type_seq_BN[:, 1:]
        seq_mask_for_loss = batch_non_pad_mask_BN[:, 1:]

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambdas_loss_samples,
            time_delta_seq=time_delta_seq_for_integral,
            seq_mask=seq_mask_for_loss,
            type_seq=type_seq_for_loss,
        )

        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events
