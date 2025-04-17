import torch
from torch import nn
import torch.nn.functional as F # Added import
import math

from easy_tpp.models.basemodel import BaseModel
from easy_tpp.config_factory import ModelConfig

class SelfCorrectingModel(BaseModel):
    """
    PyTorch implementation of the Self-Correcting Point Process model.
    Intensity for type i: lambda_i(t) = exp(mu_i + alpha_i * (t - N_i(t)))
    where N_i(t) is the number of events of type i occurred strictly before time t.
    Inherits from BaseModel.
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        """
        Initialize the Self-Correcting model.

        Args:
            model_config (EasyTPP.ModelConfig): Configuration object containing model specs.
                Expected specs:
                - 'mu' (list or tensor): Base log-intensity parameter for each type.
                - 'alpha' (list or tensor): Correction factor for each type (often negative).
        """
        super().__init__(model_config, **kwargs)

        if 'mu' not in model_config.specs or 'alpha' not in model_config.specs:
             raise ValueError("SelfCorrecting model requires 'mu' and 'alpha' in model_config.specs")

        # Convert parameters to tensors and move to the correct device
        mu = torch.tensor(model_config.specs['mu'], dtype=torch.float32)
        alpha = torch.tensor(model_config.specs['alpha'], dtype=torch.float32)

        if mu.shape[0] != self.num_event_types or alpha.shape[0] != self.num_event_types:
            raise ValueError(f"SelfCorrecting parameter dimension mismatch. Expected mu/alpha: ({self.num_event_types},). "
                             f"Got mu: {mu.shape}, alpha: {alpha.shape}")

        # Register parameters as buffers (non-trainable)
        self.register_buffer('mu', mu)     # Shape [D]
        self.register_buffer('alpha', alpha) # Shape [D]

    def _compute_N_t(self,
                     time_seq: torch.Tensor,
                     type_seq: torch.Tensor,
                     query_times: torch.Tensor):
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
        _ , seq_len_query = query_times.shape
        num_types = self.num_event_types
        device = self.device

        # Expand dimensions for broadcasting
        query_times_exp = query_times.unsqueeze(-1).to(device) # [B, L_query, 1]
        time_seq_exp = time_seq.unsqueeze(1).to(device)         # [B, 1, L_hist]
        type_seq_exp = type_seq.unsqueeze(1).to(device)         # [B, 1, L_hist]

        # Create mask for events happening *before* query times
        # Shape: [B, L_query, L_hist]
        before_query_mask = (time_seq_exp < query_times_exp) & (type_seq_exp != self.pad_token_id)

        # Create one-hot encoding for event types in history
        # type_seq_exp: [B, 1, L_hist] -> one_hot -> [B, 1, L_hist, D] (D = num_event_types)
        # Note: Need to handle pad_token_id if it's outside the range [0, D-1]
        # Assuming types are 0 to D-1. If pad_token_id is large, clamp or handle separately.
        safe_type_seq = type_seq_exp.clone()
        pad_mask = (safe_type_seq == self.pad_token_id)
        safe_type_seq[pad_mask] = 0 # Temporarily set pad to 0 for one_hot
        type_one_hot = F.one_hot(safe_type_seq.long(), num_classes=self.num_event_types).float() # [B, 1, L_hist, D]
        type_one_hot[pad_mask.unsqueeze(-1).expand_as(type_one_hot)] = 0 # Zero out one-hot for padded events

        # Combine masks: count event k if it's before query time t and has type i
        # before_query_mask: [B, L_query, L_hist, 1]
        # type_one_hot:      [B, 1,       L_hist, D]
        # Combined mask shape: [B, L_query, L_hist, D]
        count_mask = before_query_mask.unsqueeze(-1) * type_one_hot

        # Sum over history dimension (L_hist) to get counts N_i(t)
        # Shape: [B, L_query, D]
        N_t = count_mask.sum(dim=2)

        return N_t

    def compute_intensities_at_times(self,
                                     time_seq: torch.Tensor,
                                     type_seq: torch.Tensor,
                                     query_times: torch.Tensor):
        """
        Computes the intensity lambda_i(t) = exp(mu_i + alpha_i * (t - N_i(t)))
        for all event types at specified query times.

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L_hist].
            type_seq (torch.Tensor): Event types [B, L_hist].
            query_times (torch.Tensor): Times at which to compute intensities. Shape [B, L_query].

        Returns:
            torch.Tensor: Intensities lambda_i(t) for each type i. Shape [B, L_query, D].
        """
        batch_size, seq_len_query = query_times.shape
        num_types = self.num_event_types
        device = self.device

        # Compute N_i(t) for all query times and types
        # N_t shape: [B, L_query, D]
        N_t = self._compute_N_t(time_seq, type_seq, query_times)

        # Get parameters mu and alpha, ensure they are on the correct device
        mu_dev = self.mu.to(device)     # [D]
        alpha_dev = self.alpha.to(device) # [D]

        # Expand query_times for element-wise calculation: [B, L_query, 1]
        query_times_exp = query_times.unsqueeze(-1).to(device)

        # Calculate the exponent term: mu_i + alpha_i * (t - N_i(t))
        # Shapes: mu_dev[None, None, :]: [1, 1, D]
        #         alpha_dev[None, None, :]: [1, 1, D]
        #         query_times_exp: [B, L_query, 1]
        #         N_t: [B, L_query, D]
        exponent = mu_dev + alpha_dev * (query_times_exp - N_t) # Shape [B, L_query, D]

        # Compute intensity: exp(exponent)
        intensities = torch.exp(exponent)

        # Ensure non-negative intensities
        intensities = torch.clamp(intensities, min=self.eps)

        return intensities

    def compute_intensities_at_sample_times(self,
                                            time_seq: torch.Tensor,
                                            time_delta_seq: torch.Tensor, # Not used directly
                                            type_seq: torch.Tensor,
                                            sample_dtimes: torch.Tensor,
                                            compute_last_step_only: bool = False,
                                            **kwargs):
        """
        Computes intensities at sampled times relative to each event in the sequence.
        Calculates lambda(t_k + delta_t) using history up to event k (time_seq[k]).

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
            query_times = time_seq[:, -1:] + sample_dtimes[:, -1:, :]
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
            query_times = time_seq.unsqueeze(-1) + sample_dtimes

            # Loop approach (can be optimized later if needed)
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
            # List of [B, N_samples, D] -> Stack -> [L, B, N_samples, D] -> Permute -> [B, L, N_samples, D]
            if all_intensities:
                 intensities = torch.stack(all_intensities, dim=0).permute(1, 0, 2, 3)
            else:
                 intensities = torch.empty((batch_size, seq_len, num_samples, self.num_event_types), device=device)

        return intensities


    def loglike_loss(self, batch: tuple) -> tuple[torch.Tensor, int]:
        """
        Compute the log-likelihood loss for the Self-Correcting process.
        log L = sum_k log(lambda_{type_k}(t_k)) - sum_i integral_0^T lambda_i(t) dt

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
        mu_dev = self.mu.to(device)
        alpha_dev = self.alpha.to(device)

        # --- 1. Compute log intensity at each event time: sum_k log(lambda_{type_k}(t_k)) ---
        # We need lambda(t_k) using history *strictly before* t_k.
        log_lambda_at_event = torch.zeros_like(time_seq, dtype=torch.float32, device=device)

        for k in range(1, seq_len): # Start from k=1 (second event)
            # History strictly before event k
            hist_time_seq_k = time_seq[:, :k]
            hist_type_seq_k = type_seq[:, :k]

            # Query time is t_k
            query_times_k = time_seq[:, k:k+1] # [B, 1]

            # Compute intensities lambda_i(t_k) for all i, using history up to k-1
            # Output: [B, 1, D]
            intensities_at_k = self.compute_intensities_at_times(hist_time_seq_k, hist_type_seq_k, query_times_k.squeeze(-1))

            # Get the intensity for the specific event type that occurred at t_k
            event_type_k = type_seq[:, k] # [B]

            # Mask for valid events (non-padding) at step k
            valid_mask_k = (event_type_k != self.pad_token_id) # [B]

            # Use gather to select the intensity corresponding to the event type
            safe_event_type_k = event_type_k.clone()
            safe_event_type_k[~valid_mask_k] = 0 # Replace pad id with 0 for gather index

            # Gather requires index shape [B, 1, 1] for dim=2 if intensities_at_k is [B, 1, D]
            # Or squeeze intensities_at_k first: [B, D] -> gather index [B, 1] for dim=1
            lambda_event_k = torch.gather(
                intensities_at_k.squeeze(1), # [B, D]
                dim=1,
                index=safe_event_type_k.unsqueeze(-1) # [B, 1]
            ).squeeze(-1) # [B]

            # Add epsilon for numerical stability before log
            log_lambda_event_k = torch.log(lambda_event_k + self.eps)

            # Apply mask to zero out log-likelihood for padded events
            log_lambda_at_event[:, k] = log_lambda_event_k * valid_mask_k

        # Sum over sequence length (k=1 to L-1)
        total_log_event_term = log_lambda_at_event[:, 1:].sum(dim=1) # [B]

        # --- 2. Compute the integral term: sum_i integral_0^T lambda_i(t) dt ---
        # This requires integrating exp(mu_i + alpha_i * (t - N_i(t))) piecewise.
        # Integral_{t_j}^{t_{j+1}} exp(mu_i + alpha_i * (t - N_i(t_j))) dt
        # = exp(mu_i - alpha_i * N_i(t_j)) * Integral_{t_j}^{t_{j+1}} exp(alpha_i * t) dt
        # = exp(mu_i - alpha_i * N_i(t_j)) * [ (1/alpha_i) * exp(alpha_i * t) ]_{t_j}^{t_{j+1}}
        # = (1/alpha_i) * exp(mu_i - alpha_i * N_i(t_j)) * [ exp(alpha_i * t_{j+1}) - exp(alpha_i * t_j) ]
        # = (1/alpha_i) * [ exp(mu_i + alpha_i * (t_{j+1} - N_i(t_j))) - exp(mu_i + alpha_i * (t_j - N_i(t_j))) ]
        # = (1/alpha_i) * [ lambda_i(t_{j+1}^-) - lambda_i(t_j^+) ]  (using N_i at the start of interval)

        # T is the time of the last non-pad event in each sequence
        last_indices = batch_non_pad_mask.sum(dim=1) - 1
        last_indices = torch.clamp(last_indices, min=0)
        T = time_seq[torch.arange(batch_size, device=device), last_indices] # [B]

        total_integral = torch.zeros(batch_size, device=device)

        # Iterate through each sequence in the batch
        for b in range(batch_size):
            # Get valid events for this sequence
            seq_mask_b = batch_non_pad_mask[b]
            time_seq_b = time_seq[b][seq_mask_b]     # [L_valid]
            type_seq_b = type_seq[b][seq_mask_b]     # [L_valid]
            T_b = T[b]

            if time_seq_b.numel() == 0: # Handle empty sequence
                continue

            integral_b = 0.0
            # Iterate through each event type
            for i in range(num_types):
                alpha_i = alpha_dev[i]
                mu_i = mu_dev[i]

                # Find times and indices for events of type i
                type_i_mask = (type_seq_b == i)
                time_seq_bi = time_seq_b[type_i_mask] # Times of type i events

                # Define interval boundaries: 0, t_i1, t_i2, ..., t_iN, T_b
                interval_times = torch.cat([torch.tensor([0.0], device=device), time_seq_bi, T_b.unsqueeze(0)])
                interval_times = torch.unique_consecutive(interval_times) # Ensure sorted unique times

                # Calculate integral over each interval [t_start, t_end]
                for j in range(len(interval_times) - 1):
                    t_start = interval_times[j]
                    t_end = interval_times[j+1]

                    if t_start >= t_end: # Skip zero-length intervals
                        continue

                    # N_i(t) is constant within (t_start, t_end)
                    # Count events of type i strictly before t_start
                    N_i_at_start = torch.sum((time_seq_b < t_start) & (type_seq_b == i)).float()

                    # Compute integral: (1/alpha_i) * [ lambda_i(t_end^-) - lambda_i(t_start^+) ]
                    # Handle alpha_i == 0 case (Poisson) separately if needed, but assume alpha_i != 0 here.
                    if abs(alpha_i.item()) < 1e-9: # Treat as Poisson if alpha is near zero
                         lambda_const = torch.exp(mu_i)
                         interval_integral = lambda_const * (t_end - t_start)
                    else:
                        term_end = torch.exp(mu_i + alpha_i * (t_end - N_i_at_start))
                        term_start = torch.exp(mu_i + alpha_i * (t_start - N_i_at_start))
                        interval_integral = (1.0 / alpha_i) * (term_end - term_start)

                    integral_b += interval_integral

            total_integral[b] = integral_b

        # --- 3. Compute total log-likelihood per sequence ---
        log_likelihood = total_log_event_term - total_integral # [B]

        # --- 4. Compute loss (negative log-likelihood sum over batch) ---
        loss = -log_likelihood.sum()

        # --- 5. Count number of valid events (excluding the first event and padding) ---
        num_events = batch_non_pad_mask[:, 1:].sum().item()
        num_events = max(num_events, 1) # Avoid division by zero

        return loss, num_events
