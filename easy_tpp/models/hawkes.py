import torch
from torch import nn
import math

from easy_tpp.models.basemodel import BaseModel
from easy_tpp.config_factory import ModelConfig

class HawkesModel(BaseModel):
    """
    PyTorch implementation of the Hawkes process model.
    Inherits from BaseModel for integration with the framework, enabling
    methods like predict_one_step_at_every_event.
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
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

        if 'mu' not in model_config.specs or 'alpha' not in model_config.specs or 'beta' not in model_config.specs:
             raise ValueError("Hawkes model requires 'mu', 'alpha', and 'beta' in model_config.specs")

        # Convert parameters to tensors and move to the correct device
        mu = torch.tensor(model_config.specs['mu'], dtype=torch.float32)
        alpha = torch.tensor(model_config.specs['alpha'], dtype=torch.float32)
        beta = torch.tensor(model_config.specs['beta'], dtype=torch.float32)

        if mu.shape[0] != self.num_event_types or \
           alpha.shape != (self.num_event_types, self.num_event_types) or \
           beta.shape != (self.num_event_types, self.num_event_types):
            raise ValueError(f"Hawkes parameter dimensions mismatch. Expected mu: ({self.num_event_types},), "
                             f"alpha/beta: ({self.num_event_types}, {self.num_event_types}). "
                             f"Got mu: {mu.shape}, alpha: {alpha.shape}, beta: {beta.shape}")

        # Register parameters as buffers (non-trainable)
        self.register_buffer('mu', mu)
        self.register_buffer('alpha', alpha)
        self.register_buffer('beta', beta)

    def compute_intensities_at_times(self,
                                     time_seq: torch.Tensor,
                                     type_seq: torch.Tensor,
                                     query_times: torch.Tensor):
        """
        Computes the intensity lambda(t) for all event types at specified query times.
        lambda_i(t) = mu_i + sum_{j=1}^{D} sum_{t_k < t, type_k=j} alpha_{ij} * exp(-beta_{ij} * (t - t_k))

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L_hist].
            type_seq (torch.Tensor): Event types [B, L_hist]. Assumes padding uses self.pad_token_id.
            query_times (torch.Tensor): Times at which to compute intensities. Shape [B, L_query].

        Returns:
            torch.Tensor: Intensities lambda_i(t) for each type i. Shape [B, L_query, D].
        """
        batch_size, seq_len_hist = time_seq.shape
        _ , seq_len_query = query_times.shape
        num_types = self.num_event_types
        device = self.device

        # Ensure query_times has the correct shape for broadcasting: [B, L_query, 1]
        query_times_exp = query_times.unsqueeze(-1).to(device) # [B, L_query, 1]

        # Expand history sequences for broadcasting with query times
        # time_seq: [B, 1, L_hist]
        # type_seq: [B, 1, L_hist]
        time_seq_exp = time_seq.unsqueeze(1).to(device)
        type_seq_exp = type_seq.unsqueeze(1).to(device)

        # Calculate time differences: delta_t = query_t - history_t
        # Shape: [B, L_query, L_hist]
        time_diffs = query_times_exp - time_seq_exp

        # Mask for valid historical events (time_k < query_t and not padding)
        valid_event_mask = (time_diffs > self.eps) & (type_seq_exp != self.pad_token_id) # [B, L_query, L_hist]

        # Gather alpha and beta based on history event types (source type k)
        # alpha[i, k], beta[i, k] where k = type_seq_exp
        safe_type_seq = type_seq_exp.clone()
        safe_type_seq[type_seq_exp == self.pad_token_id] = 0 # Replace pad id with a valid index (e.g., 0)

        # Index alpha and beta: alpha[target_type, source_type]
        # alpha is [D, D]. We need alpha[:, safe_type_seq] -> [D, B, 1, L_hist]
        alpha_indexed = self.alpha[:, safe_type_seq]
        beta_indexed = self.beta[:, safe_type_seq]

        # Transpose to [B, 1, L_hist, D] for broadcasting with decay
        alpha_gathered = alpha_indexed.permute(1, 2, 3, 0) # [B, 1, L_hist, D]
        beta_gathered = beta_indexed.permute(1, 2, 3, 0)  # [B, 1, L_hist, D]

        # Calculate exponential decay term: exp(-beta * delta_t)
        # beta_gathered: [B, 1, L_hist, D]
        # time_diffs: [B, L_query, L_hist] -> unsqueeze(-1) -> [B, L_query, L_hist, 1]
        exp_decay = torch.exp(-beta_gathered * time_diffs.unsqueeze(-1)) # [B, L_query, L_hist, D]

        # Calculate contribution from each historical event: alpha * exp_decay
        # Shape: [B, L_query, L_hist, D]
        event_contributions = alpha_gathered * exp_decay

        # Apply mask to zero out contributions from invalid events (padding or future events)
        # valid_event_mask: [B, L_query, L_hist] -> unsqueeze(-1) -> [B, L_query, L_hist, 1]
        masked_contributions = event_contributions * valid_event_mask.unsqueeze(-1)

        # Sum contributions over the history dimension (L_hist)
        # Shape: [B, L_query, D]
        summed_contributions = masked_contributions.sum(dim=2)

        # Add the base intensity mu
        # mu: [D] -> broadcastable to [B, L_query, D]
        intensities = self.mu.to(device) + summed_contributions # [B, L_query, D]

        # Ensure non-negative intensities
        intensities = torch.clamp(intensities, min=self.eps)

        return intensities


    def compute_intensities_at_sample_times(self,
                                            time_seq: torch.Tensor,
                                            time_delta_seq: torch.Tensor, # Not directly used here
                                            type_seq: torch.Tensor,
                                            sample_dtimes: torch.Tensor,
                                            compute_last_step_only: bool = False,
                                            **kwargs):
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

        time_seq = time_seq.to(device)
        type_seq = type_seq.to(device)
        sample_dtimes = sample_dtimes.to(device)

        if compute_last_step_only:
            # History is the full sequence
            hist_time_seq = time_seq
            hist_type_seq = type_seq
            # Query times are relative to the *last* event time
            query_times = time_seq[:, -1:] + sample_dtimes[:, -1:, :] # [B, 1, N_samples]
            query_times = query_times.squeeze(1) # Shape [B, N_samples]

            # Compute intensities at these query times using the full history
            # Output shape: [B, N_samples, D]
            intensities = self.compute_intensities_at_times(hist_time_seq, hist_type_seq, query_times)
            # Reshape to match expected output: [B, 1, N_samples, D]
            intensities = intensities.unsqueeze(1)

        else:
            # Compute for every step in the sequence
            # Query times are t_k + delta_t_n for each k and sample n
            # query_times shape: [B, L, N_samples]
            query_times = time_seq.unsqueeze(-1) + sample_dtimes # Broadcasting t_k to all samples

            # We need to compute intensity at query_times[b, k, n] using history up to time_seq[b, k].
            # Loop approach for clarity, might be slow for very long sequences.
            all_intensities = []
            for k in range(seq_len):
                # History up to event k (inclusive)
                hist_time_seq_k = time_seq[:, :k+1]
                hist_type_seq_k = type_seq[:, :k+1]

                # Query times relative to event k
                query_times_k = query_times[:, k, :] # [B, N_samples]

                # Compute intensities at these times using history up to k
                # Output: [B, N_samples, D]
                intensities_k = self.compute_intensities_at_times(hist_time_seq_k, hist_type_seq_k, query_times_k)
                all_intensities.append(intensities_k)

            # Stack results along the sequence length dimension (L)
            # List of tensors [B, N_samples, D] -> Stack -> [L, B, N_samples, D] -> Permute -> [B, L, N_samples, D]
            if all_intensities:
                 intensities = torch.stack(all_intensities, dim=0).permute(1, 0, 2, 3)
            else:
                 # Handle empty sequence case if necessary
                 intensities = torch.empty((batch_size, seq_len, num_samples, self.num_event_types), device=device)


        return intensities


    def loglike_loss(self, batch: tuple) -> tuple[torch.Tensor, int]:
        """
        Compute the log-likelihood loss for the Hawkes process.
        log L = sum_k log(lambda_{type_k}(time_k)) - sum_i integral_0^T lambda_i(t) dt

        Args:
            batch: Tuple containing time_seq, time_delta_seq, type_seq, batch_non_pad_mask, _

        Returns:
            tuple: (total negative log-likelihood loss, number of events)
        """
        time_seq, time_delta_seq, type_seq, batch_non_pad_mask, _ = batch
        batch_size, seq_len = time_seq.shape
        num_types = self.num_event_types
        device = self.device

        time_seq = time_seq.to(device)
        type_seq = type_seq.to(device)
        batch_non_pad_mask = batch_non_pad_mask.to(device)

        # 1. Compute log intensity at each event time: sum_k log(lambda_{type_k}(time_k))
        # We need lambda(t_k) using history *before* t_k.
        log_lambda_at_event = torch.zeros_like(time_seq, dtype=torch.float32, device=device)

        # Compute intensities for all events k=1...L-1 at once if possible?
        # Requires computing lambda(t_k) using history up to k-1.
        # The loop approach might be clearer here as well.
        for k in range(1, seq_len): # Start from k=1 (second event)
            # History strictly before event k
            hist_time_seq_k = time_seq[:, :k]
            hist_type_seq_k = type_seq[:, :k]

            # Query time is t_k
            query_times_k = time_seq[:, k:k+1].squeeze(-1) # [B]

            # Compute intensities lambda_i(t_k) for all i, using history up to k-1
            # Output: [B, 1, D] -> squeeze -> [B, D]
            intensities_at_k = self.compute_intensities_at_times(hist_time_seq_k, hist_type_seq_k, query_times_k).squeeze(1)

            # Get the intensity for the specific event type that occurred at t_k
            event_type_k = type_seq[:, k] # [B]

            # Mask for valid events (non-padding) at step k
            valid_mask_k = (event_type_k != self.pad_token_id) # [B]

            # Use gather to select the intensity corresponding to the event type
            safe_event_type_k = event_type_k.clone()
            safe_event_type_k[~valid_mask_k] = 0 # Replace pad id with 0 for gather index

            # Gather requires index shape [B, 1] for dim=1
            lambda_event_k = torch.gather(intensities_at_k, dim=1, index=safe_event_type_k.unsqueeze(-1)).squeeze(-1) # [B]

            # Add epsilon for numerical stability before log
            log_lambda_event_k = torch.log(lambda_event_k + self.eps)

            # Apply mask to zero out log-likelihood for padded events
            log_lambda_at_event[:, k] = log_lambda_event_k * valid_mask_k

        # Sum over sequence length (k=1 to L-1)
        total_log_event_term = log_lambda_at_event[:, 1:].sum(dim=1) # [B]

        # 2. Compute the integral term: sum_i integral_0^T lambda_i(t) dt (Analytic calculation)
        # Integral = sum_i mu_i * T + sum_{i,j} sum_{t_k < T, type_k=j} (alpha_{ij} / beta_{ij}) * (1 - exp(-beta_{ij} * (T - t_k)))

        # T is the time of the last non-pad event in each sequence
        last_indices = batch_non_pad_mask.sum(dim=1) - 1
        # Handle sequences with zero length (all padding)
        last_indices = torch.clamp(last_indices, min=0)
        T = time_seq[torch.arange(batch_size, device=device), last_indices] # [B]
        T = T.unsqueeze(-1) # [B, 1] for broadcasting

        # Base rate integral: sum_i mu_i * T
        integral_base = self.mu.sum() * T.squeeze(-1) # [B]

        # Event contribution integral
        hist_times = time_seq # [B, L]
        hist_types = type_seq # [B, L]

        # Time differences: T - t_k
        time_diffs_T = T - hist_times # [B, L]

        # Mask for valid events contributing to the integral: time_k < T and not padding
        # Use batch_non_pad_mask to only consider actual events in the sequence for the integral calculation
        valid_integral_mask = (time_diffs_T > self.eps) & batch_non_pad_mask # [B, L]

        # Gather alpha[i, type_k] and beta[i, type_k] for all target types i
        safe_hist_types = hist_types.clone()
        safe_hist_types[hist_types == self.pad_token_id] = 0 # Replace pad id with 0

        # alpha_for_hist[i, b, k] = alpha[i, hist_types[b, k]]
        # beta_for_hist[i, b, k] = beta[i, hist_types[b, k]]
        alpha_for_hist = self.alpha[:, safe_hist_types] # [D, B, L]
        beta_for_hist = self.beta[:, safe_hist_types]   # [D, B, L]

        # Calculate term for each (i, b, k)
        # Add eps to beta to avoid division by zero, though beta should ideally be > 0
        alpha_over_beta = alpha_for_hist / (beta_for_hist + self.eps) # [D, B, L]
        # Unsqueeze time_diffs_T for broadcasting with beta_for_hist
        exp_term = torch.exp(-beta_for_hist * time_diffs_T.unsqueeze(0)) # [D, B, L]
        decay_term = 1.0 - exp_term # [D, B, L]

        contribution_ijk = alpha_over_beta * decay_term # [D, B, L]

        # Apply mask: zero out contributions from invalid events (k)
        # Unsqueeze valid_integral_mask for broadcasting
        masked_contribution_ijk = contribution_ijk * valid_integral_mask.unsqueeze(0) # [D, B, L]

        # Sum over target types (i) and history events (k) for each batch element (b)
        integral_event = masked_contribution_ijk.sum(dim=[0, 2]) # Sum over D and L -> [B]

        # Total integral
        total_integral = integral_base + integral_event # [B]

        # 3. Compute total log-likelihood per sequence
        # Note: The first event t_0 doesn't contribute to the event sum term.
        log_likelihood = total_log_event_term - total_integral # [B]

        # 4. Compute loss (negative log-likelihood sum over batch)
        loss = -log_likelihood.sum()

        # 5. Count number of valid events (excluding the first event and padding) used in the loss calculation
        # Events from index 1 onwards that are not padding contribute to the log_lambda term.
        num_events = batch_non_pad_mask[:, 1:].sum().item()
        # Avoid division by zero if num_events is 0
        num_events = max(num_events, 1)


        return loss, num_events

