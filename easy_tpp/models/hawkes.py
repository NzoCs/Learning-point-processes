import torch

from easy_tpp.configs.model_config import ModelConfig
from easy_tpp.models.basemodel import BaseModel


class Hawkes(BaseModel):
    """
    PyTorch implementation of the Hawkes process model.
    Inherits from BaseModel for integration with the framework, enabling
    methods like predict_one_step_at_every_event.
    """

    def __init__(self, model_config: ModelConfig, **kwargs) -> None:
        """
        Initialize the Hawkes model.

        Args:
            model_config (EasyTPP.ModelConfig): Configuration object containing model specs.
                Expected specs: 'mu' (list), 'alpha' (list of lists), 'beta' (list of lists).
        """
        super().__init__(model_config, **kwargs)

        # Load Hawkes parameters from config
        # Ensure they are tensors on the correct device
        # mu: [num_event_types]
        # alpha: [num_event_types, num_event_types] (alpha[i, j] effect of type j on type i)
        # beta: [num_event_types, num_event_types] (beta[i, j] decay rate for effect of type j on type i)

        # Convert parameters to tensors and move to the correct device
        mu = torch.tensor(model_config.specs.mu, dtype=torch.float32).view(
            self.num_event_types
        )
        alpha = torch.tensor(model_config.specs.alpha, dtype=torch.float32).view(
            self.num_event_types, self.num_event_types
        )
        beta = torch.tensor(model_config.specs.beta, dtype=torch.float32).view(
            self.num_event_types, self.num_event_types
        )

        if (
            mu.shape[0] != self.num_event_types
            or alpha.shape != (self.num_event_types, self.num_event_types)
            or beta.shape != (self.num_event_types, self.num_event_types)
        ):
            raise ValueError(
                f"Hawkes parameter dimensions mismatch. Expected mu: ({self.num_event_types},), "
                f"alpha/beta: ({self.num_event_types}, {self.num_event_types}). "
                f"Got mu: {mu.shape}, alpha: {alpha.shape}, beta: {beta.shape}"
            )

        # Ensure beta values are positive for numerical stability
        beta = torch.clamp(beta, min=self.eps)

        # Register parameters as buffers (non-trainable)
        self.register_buffer("mu", mu)
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)

    def compute_intensities_at_times(
        self,
        time_seq: torch.Tensor,
        time_delta_seq: torch.Tensor,  # Not directly used here
        type_seq: torch.Tensor,
        query_times: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the intensity lambda(t) for all event types at specified query times.
        lambda_i(t) = mu_i + sum_{j=1}^{D} sum_{t_k < t, type_k=j} alpha_{ij} * exp(-beta_{ij} * (t - t_k))

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L_hist].
            type_seq (torch.Tensor): Event types [B, L_hist]. Assumes padding uses self.pad_token_id.
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

        # Reshape for broadcasting
        # query_times: [B, L_query, N_samples] -> [B, L_query, N_samples, 1]
        query_times_exp = query_times.unsqueeze(-1).to(device)

        # time_seq: [B, L_hist] -> [B, 1, 1, L_hist]
        time_seq_exp = time_seq.unsqueeze(1).unsqueeze(2).to(device)

        # type_seq: [B, L_hist] -> [B, 1, 1, L_hist]
        type_seq_exp = type_seq.unsqueeze(1).unsqueeze(2).to(device)

        # Calculate time differences: delta_t = query_t - history_t
        # Shape: [B, L_query, N_samples, L_hist]
        time_diffs = query_times_exp - time_seq_exp

        # Mask for valid historical events (time_k < query_t and not padding)
        # Shape: [B, L_query, N_samples, L_hist]
        valid_event_mask = (time_diffs >= 0) & (type_seq_exp != self.pad_token_id)

        # Replace pad_token_id with a valid index (0) to avoid indexing errors
        safe_type_seq = type_seq_exp.clone()
        safe_type_seq[type_seq_exp == self.pad_token_id] = 0

        # Index alpha and beta: alpha[target_type, source_type]
        # alpha is [D, D]. We need alpha[:, safe_type_seq] -> [D, B, 1, 1, L_hist]
        alpha_indexed = self.alpha[:, safe_type_seq]
        beta_indexed = self.beta[:, safe_type_seq]

        # Transpose to [B, L_query, N_samples, L_hist, D] for broadcasting with decay
        alpha_gathered = alpha_indexed.permute(1, 2, 3, 4, 0)
        beta_gathered = beta_indexed.permute(1, 2, 3, 4, 0)

        # Calculate exponential decay term: exp(-beta * delta_t)
        # beta_gathered: [B, L_query, N_samples, L_hist, D]
        # time_diffs: [B, L_query, N_samples, L_hist] -> unsqueeze(-1) -> [B, L_query, N_samples, L_hist, 1]
        exp_decay = torch.exp(-beta_gathered * time_diffs.unsqueeze(-1))

        # Calculate contribution from each historical event: alpha * exp_decay
        # Shape: [B, L_query, N_samples, L_hist, D]
        event_contributions = alpha_gathered * exp_decay

        # Apply mask to zero out contributions from invalid events (padding or future events)
        # valid_event_mask: [B, L_query, N_samples, L_hist] -> unsqueeze(-1) -> [B, L_query, N_samples, L_hist, 1]
        masked_contributions = event_contributions * valid_event_mask.unsqueeze(-1)
        masked_contributions = torch.nan_to_num(
            masked_contributions, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Sum contributions over the history dimension (L_hist)
        # Shape: [B, L_query, N_samples, D]
        summed_contributions = masked_contributions.sum(dim=3)

        # Add the base intensity mu
        # mu: [D] -> broadcastable to [B, L_query, N_samples, D]
        intensities = self.mu.to(device) + summed_contributions

        # Ensure non-negative intensities
        intensities = torch.clamp(intensities, min=self.eps)

        # Remove N_samples dimension if it wasn't in the input
        if not has_samples:
            intensities = intensities.squeeze(2)

        return intensities

    def compute_intensities_at_sample_times(
        self,
        time_seq: torch.Tensor,
        time_delta_seq: torch.Tensor,
        type_seq: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes intensities at sampled times relative to each event in the sequence.
        Required by BaseModel for prediction and loss calculation.
        Calculates lambda(t_k + delta_t) using history up to event k (time_seq[k]).

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L].
            time_delta_seq (torch.Tensor): Time differences between events [B, L].
            type_seq (torch.Tensor): Event types [B, L].
            sample_dtimes (torch.Tensor): Sampled time deltas relative to each event [B, L, N_samples].
            compute_last_step_only (bool): If True, only compute for the last event in the sequence.

        Returns:
            torch.Tensor: Intensities lambda_i(t_k + delta_t). Shape [B, L, N_samples, D] or [B, 1, N_samples, D] if compute_last_step_only.
        """
        batch_size, seq_len = time_seq.shape
        num_samples = sample_dtimes.shape[-1]
        device = self.device

        num_samples = sample_dtimes.shape[-1] if sample_dtimes.dim() == 3 else 1

        if compute_last_step_only:
            time_seq_clone = time_seq[:, -2:]
            safe_dtimes = time_seq_clone.diff(dim=1).unsqueeze(-1)  # [B, 1, 1]
            ratios = torch.linspace(0, 1, num_samples, device=device).view(
                1, 1, num_samples
            )  # [1, 1, N_samples]
            query_times = time_seq + safe_dtimes * ratios

        else:
            time_seq_clone = time_seq
            safe_dtimes = time_seq_clone.diff(dim=1).unsqueeze(-1)  # [B, L-1, 1]
            ratios = torch.linspace(0, 1, num_samples, device=device).view(
                1, 1, num_samples
            )  # [1, 1, N_samples]
            query_times = time_seq[:, :-1] + safe_dtimes * ratios

        intensities = self.compute_intensities_at_times(
            time_seq=time_seq[:, :-1],
            time_delta_seq=None,
            type_seq=type_seq[:, :-1],
            query_times=query_times,
        )

        return intensities

    def loglike_loss(
        self,
        batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, int]:
        """Compute the log-likelihood loss for the Hawkes model.

        Args:
            batch (tuple): A tuple containing time_seq, time_delta_seq, type_seq,
                           batch_non_pad_mask, and batch_attention_mask.

        Returns:
            tuple: loss (torch.Tensor), num_events (int).
        """
        time_seq_BN, time_delta_seq_BN, type_seq_BN, batch_non_pad_mask_BN, _ = batch

        # For lambda_at_event: intensity at actual event times t_1, ..., t_N
        # query_times_for_event_ll: [B, L-1, 1] where L is seq_len
        # These are absolute timestamps t_1, ..., t_N
        query_times_for_event_ll = time_seq_BN[:, 1:].unsqueeze(-1)

        # lambda_at_event: Intensities at events t_1, ..., t_N. Shape: [B, L-1, D]
        # History for these queries is the full time_seq_BN, type_seq_BN
        lambda_at_event = self.compute_intensities_at_times(
            time_seq=time_seq_BN,
            time_delta_seq=None,
            type_seq=type_seq_BN,
            query_times=query_times_for_event_ll,
        ).squeeze(-2)

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
