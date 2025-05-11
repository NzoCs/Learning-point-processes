import torch

from easy_tpp.models.basemodel import BaseModel
from easy_tpp.config_factory.model_config import ModelConfig

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

        # Convert parameters to tensors and move to the correct device
        mu = torch.tensor(model_config.specs.mu, dtype=torch.float32).view(self.num_event_types)
        alpha = torch.tensor(model_config.specs.alpha, dtype=torch.float32).view(self.num_event_types, self.num_event_types)
        beta = torch.tensor(model_config.specs.beta, dtype=torch.float32).view(self.num_event_types, self.num_event_types)

        if mu.shape[0] != self.num_event_types or \
           alpha.shape != (self.num_event_types, self.num_event_types) or \
           beta.shape != (self.num_event_types, self.num_event_types):
            raise ValueError(f"Hawkes parameter dimensions mismatch. Expected mu: ({self.num_event_types},), "
                             f"alpha/beta: ({self.num_event_types}, {self.num_event_types}). "
                             f"Got mu: {mu.shape}, alpha: {alpha.shape}, beta: {beta.shape}")

        # Ensure beta values are positive for numerical stability
        beta = torch.clamp(beta, min=self.eps)

        # Register parameters as buffers (non-trainable)
        self.register_buffer('mu', mu)
        self.register_buffer('alpha', alpha)
        self.register_buffer('beta', beta)


    def compute_intensities_at_times(self,
                                     time_seq: torch.Tensor,
                                     time_delta_seq: torch.Tensor, # Not directly used here
                                     type_seq: torch.Tensor,
                                     query_times: torch.Tensor,
                                     **kwargs):
        """
        Computes the intensity lambda(t) for all event types at specified query times.
        lambda_i(t) = mu_i + sum_{j=1}^{D} sum_{t_k < t, type_k=j} alpha_{ij} * exp(-beta_{ij} * (t - t_k))

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L_hist].
            type_seq (torch.Tensor): Event types [B, L_hist]. Assumes padding uses self.pad_token_id.
            query_times (torch.Tensor): Times at which to compute intensities. Shape [B, L_query, N_samples].

        Returns:
            torch.Tensor: Intensities lambda_i(t) for each type i. Shape [B, L_query, N_samples, D].
        """

        batch_size, seq_len = time_seq.shape
        num_samples = query_times.shape[-1]
        device = self.device


        # Ensure query_times has the correct shape for broadcasting: [B, L_query, 1]
        query_times = query_times.to(device) # [B, L_query, 1]

        # 2. Create a causal mask to ensure each position only sees its history
        # First reshape time_seq for broadcasting: [B, L, 1] (event times)
        time_seq_expanded = time_seq.unsqueeze(-1)
        
        # Create position indices matrix [1...L]
        seq_positions = torch.arange(1, seq_len+1, device=device).view(1, -1, 1)
        history_positions = torch.arange(1, seq_len+1, device=device).view(1, 1, -1)
        
        # Causal mask: history_positions ≤ seq_positions
        # Shape: [1, L, L] where True means this history position is valid for this sequence position
        causal_mask = history_positions <= seq_positions
        
        # 3. Reshape for broadcasting with causal mask
        batch_indices = torch.arange(batch_size, device=device)
        
        # 4. For each position k and sample n, compute intensity using history up to k
        # Reshape query_times: [B, L, N_samples] → [B, L, N_samples, 1]
        query_times_expanded = query_times.unsqueeze(-1)
        
        # 5. Expand time and type sequences for vectorized computation
        # [B, L] → [B, 1, 1, L] ready for broadcasting
        time_seq_hist = time_seq.unsqueeze(1).unsqueeze(1)  
        type_seq_hist = type_seq.unsqueeze(1).unsqueeze(1)
        
        # 6. Create valid event mask using causal mask
        # [B, 1, 1, L] & [1, L, 1, L] → [B, L, N_samples, L]
        time_diffs = query_times_expanded - time_seq_hist
        pad_mask = (type_seq_hist != self.pad_token_id)
        
        # Fix: correctly expand masks without adding extra dimensions
        # Shape: [1, L, L] -> [1, L, 1, L] -> [B, L, N_samples, L]
        causal_mask_expanded = causal_mask.unsqueeze(2).expand(batch_size, seq_len, num_samples, seq_len)
        pad_mask_expanded = pad_mask.expand(batch_size, seq_len, num_samples, seq_len)
        
        # True where: event is before query time AND is in history of position k AND is not padding
        valid_mask = (time_diffs > self.eps) & causal_mask_expanded & pad_mask_expanded
        
        # 7. Handle event types safely
        safe_type_seq = torch.clamp(
            type_seq_hist.clone().masked_fill_(type_seq_hist == self.pad_token_id, 0),
            0, self.num_event_types - 1
        )
        
        # 8. Gather alpha and beta matrices based on history event types
        # Use advanced indexing to select appropriate parameters
        # First get alpha/beta for each event type
        alpha_indexed = self.alpha[:, safe_type_seq.squeeze(1)]  # [D, B, 1, L]
        beta_indexed = self.beta[:, safe_type_seq.squeeze(1)]    # [D, B, 1, L]
        
        # 9. Fix: correctly reshape and expand alpha/beta tensors
        # First reshape to add seq_len dimension: [D, B, 1, L] -> [D, B, 1, 1, L]
        alpha_reshaped = alpha_indexed.unsqueeze(2)
        beta_reshaped = beta_indexed.unsqueeze(2)
        
        # Permute to [B, 1, 1, D, L]
        alpha_permuted = alpha_reshaped.permute(1, 2, 3, 0, 4)
        beta_permuted = beta_reshaped.permute(1, 2, 3, 0, 4)
        
        # Now expand to [B, L, N, D, L]
        alpha_expanded = alpha_permuted.expand(batch_size, seq_len, num_samples, self.num_event_types, seq_len)
        beta_expanded = beta_permuted.expand(batch_size, seq_len, num_samples, self.num_event_types, seq_len)
        
        # 10. Compute exponential decay term for all positions at once
        time_diffs_expanded = time_diffs.expand(batch_size, seq_len, num_samples, seq_len)
        exp_decay = torch.exp(-beta_expanded * time_diffs_expanded.unsqueeze(-2))
        
        # 11. Calculate contribution from each historical event
        event_contributions = alpha_expanded * exp_decay
        
        # 12. Apply mask to valid events only
        masked_contributions = event_contributions * valid_mask.unsqueeze(-2)
        
        # 13. Sum contributions over the history dimension (L)
        summed_contributions = masked_contributions.sum(dim=-1)
        
        # 14. Add the base intensity mu
        # Expand mu: [D] → [1, 1, 1, D]
        mu_expanded = self.mu.view(1, 1, 1, -1).expand(batch_size, seq_len, num_samples, -1)
        intensities = mu_expanded + summed_contributions
        
        # 15. Ensure non-negative intensities
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
            # Query times are relative to the *last* event time
            time_seq = time_seq[:, -1:]
            sample_dtimes = sample_dtimes[:, -1:, :]
            
        sample_dt_exp = sample_dtimes.unsqueeze(-1).expand(-1, -1, -1, self.num_event_types)
        mask = (type_seq.view(batch_size, seq_len, 1, 1) == torch.arange(self.num_event_types).view(1,1,1,self.num_event_types)).expand(-1, -1, num_samples, -1)
        masked_sample_dt = torch.where(mask, sample_dt_exp, 0)
        sample_times = masked_sample_dt.cumsum(dim=1)
        query_times = torch.zeros_like(sample_dtimes)

        for j in range(self.num_event_types) :
            query_times +=  torch.where(mask[..., j], sample_times[..., j], 0)

        # Compute intensities at these query times using the full history
        # Output shape: [B, L, N_samples, D]
        intensities = self.compute_intensities_at_times(time_seq, None, type_seq, query_times)

        return intensities
        
    def loglike_loss(self, batch):
        """Compute the log-likelihood loss for the Hawkes model.

        Args:
            batch (tuple): A tuple containing time_seq, time_delta_seq, type_seq,
                           batch_non_pad_mask, and batch_attention_mask.

        Returns:
            tuple: loss (torch.Tensor), num_events (int).
        """
        time_seq_BN, time_delta_seq_BN, type_seq_BN, batch_non_pad_mask_BN, _ = batch

        # Slice sequences for log-likelihood computation (N-1 intervals)
        # We are interested in lambda(t_i) for events t_1, ..., t_N
        # and integrals over (t_0, t_1), ..., (t_{N-1}, t_N)
        # So, most sequences will be of length N (or N-1 if we consider N intervals)

        # For lambda_at_event (intensity at t_i), query times are t_1, ..., t_N. History is up to t_N.
        # time_seq_BN itself is [t_0, ..., t_N]
        # query_times for lambda_at_event should be t_1, ..., t_N.
        # These are time_seq_BN[:, 1:]
        query_times_for_event_ll = time_seq_BN[:, :].unsqueeze(-1)  # [B, N-1, 1] if N is original seq_len+1
                                                                    # or [B, N, 1] if N is original seq_len

        # lambda_at_event: Intensities at actual event times t_1, ..., t_N (or t_0,...,t_{N-1} depending on convention)
        # NHP uses left_hiddens for events t_1, ..., t_N. So seq_len-1.
        # Let's align with NHP: consider N-1 events for log-likelihood (events at t_1 to t_N)
        # time_seq_BN is [B, L], type_seq_BN is [B, L]
        # query_times_for_event_ll will be [B, L-1, 1] using time_seq_BN[:, 1:]
        # The history for these queries is time_seq_BN, type_seq_BN (full history)
        lambda_at_event = self.compute_intensities_at_times(
            time_seq=time_seq_BN,
            time_delta_seq=None,  # Not used by this specific Hawkes intensity function
            type_seq=type_seq_BN,
            query_times=query_times_for_event_ll
        )
        lambda_at_event = lambda_at_event.squeeze(-2)  # Shape: [B, L-1, D]

        # For integral term, consider intervals (t_0,t_1), ..., (t_{L-1}, t_L)
        # time_delta_seq_BN[:, 1:] gives dt_1, ..., dt_L. Shape [B, L-1]
        time_delta_seq_N_minus_1 = time_delta_seq_BN[:, :]

        # dts_samples_for_integral: Samples within each interval dt_1, ..., dt_L
        # Shape: [B, L-1, G]
        dts_samples_for_integral = self.make_dtime_loss_samples(time_delta_seq_N_minus_1)

        # lambdas_loss_samples: Intensities at these sampled times
        # History for interval (t_k, t_{k+1}) is events up to t_k.
        # So, time_seq and type_seq for compute_intensities_at_sample_times should be t_0, ..., t_{L-1}
        # This corresponds to time_seq_BN[:, :-1]
        time_seq_hist_for_integral = time_seq_BN[:, :]
        type_seq_hist_for_integral = type_seq_BN[:, :]

        lambdas_loss_samples = self.compute_intensities_at_sample_times(
            time_seq=time_seq_hist_for_integral,
            time_delta_seq=None, # Not used by this specific Hawkes intensity function
            type_seq=type_seq_hist_for_integral,
            sample_dtimes=dts_samples_for_integral, # These are deltas from t_k
            compute_last_step_only=False
        ) # Shape: [B, L-1, G, D]

        # Prepare other arguments for compute_loglikelihood
        type_seq_N_minus_1 = type_seq_BN[:, :] # Types of events t_1, ..., t_L
        seq_mask_N_minus_1 = batch_non_pad_mask_BN[:, :] # Mask for events t_1, ..., t_L

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambdas_loss_samples,
            time_delta_seq=time_delta_seq_N_minus_1,
            seq_mask=seq_mask_N_minus_1,
            type_seq=type_seq_N_minus_1
        )

        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events