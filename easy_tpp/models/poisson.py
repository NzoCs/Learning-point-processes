import torch
from torch import nn
import math

from easy_tpp.models.basemodel import BaseModel
from easy_tpp.config_factory import ModelConfig

class PoissonModel(BaseModel):
    """
    PyTorch implementation of the Homogeneous Poisson process model.
    Assumes a constant intensity `mu_i` for each event type `i`.
    Inherits from BaseModel.
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        """
        Initialize the Homogeneous Poisson model.

        Args:
            model_config (EasyTPP.ModelConfig): Configuration object containing model specs.
                Expected specs: 'mu' (list or tensor of base intensities for each event type).
        """
        super().__init__(model_config, **kwargs)

        if 'mu' not in model_config.specs:
             raise ValueError("Poisson model requires 'mu' in model_config.specs")

        # Convert mu to a tensor and move to the correct device
        mu = torch.tensor(model_config.specs['mu'], dtype=torch.float32)

        if mu.shape[0] != self.num_event_types:
            raise ValueError(f"Poisson parameter dimension mismatch. Expected mu: ({self.num_event_types},). "
                             f"Got mu: {mu.shape}")

        # Register mu as a buffer (non-trainable parameter)
        self.register_buffer('mu', mu) # Shape [D]

    def compute_intensities_at_times(self,
                                     time_seq: torch.Tensor, # Not used
                                     type_seq: torch.Tensor, # Not used
                                     query_times: torch.Tensor):
        """
        Computes the constant intensity lambda(t) = mu for all event types at specified query times.

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L_hist]. (Ignored)
            type_seq (torch.Tensor): Event types [B, L_hist]. (Ignored)
            query_times (torch.Tensor): Times at which to compute intensities. Shape [B, L_query].

        Returns:
            torch.Tensor: Constant intensities mu_i for each type i. Shape [B, L_query, D].
        """
        batch_size, seq_len_query = query_times.shape
        num_types = self.num_event_types
        device = self.device

        # Expand mu to match the query dimensions
        # mu: [D] -> [1, 1, D] -> broadcast to [B, L_query, D]
        intensities = self.mu.view(1, 1, num_types).expand(batch_size, seq_len_query, num_types).to(device)

        # Ensure non-negative intensities (although mu should be positive)
        intensities = torch.clamp(intensities, min=self.eps)

        return intensities

    def compute_intensities_at_sample_times(self,
                                            time_seq: torch.Tensor, # Not used
                                            time_delta_seq: torch.Tensor, # Not used
                                            type_seq: torch.Tensor, # Not used
                                            sample_dtimes: torch.Tensor,
                                            compute_last_step_only: bool = False,
                                            **kwargs):
        """
        Computes constant intensities mu at sampled times.

        Args:
            time_seq (torch.Tensor): Event timestamps [B, L]. (Ignored)
            time_delta_seq (torch.Tensor): Time differences [B, L]. (Ignored)
            type_seq (torch.Tensor): Event types [B, L]. (Ignored)
            sample_dtimes (torch.Tensor): Sampled time deltas [B, L, N_samples].
            compute_last_step_only (bool): If True, adjusts output shape.

        Returns:
            torch.Tensor: Constant intensities mu_i. Shape [B, L, N_samples, D] or [B, 1, N_samples, D] if compute_last_step_only.
        """
        batch_size, seq_len, num_samples = sample_dtimes.shape
        num_types = self.num_event_types
        device = self.device

        # Expand mu to match the sample dimensions
        # mu: [D] -> [1, 1, 1, D] -> broadcast
        if compute_last_step_only:
            # Target shape: [B, 1, N_samples, D]
            intensities = self.mu.view(1, 1, 1, num_types).expand(batch_size, 1, num_samples, num_types).to(device)
        else:
            # Target shape: [B, L, N_samples, D]
            intensities = self.mu.view(1, 1, 1, num_types).expand(batch_size, seq_len, num_samples, num_types).to(device)

        # Ensure non-negative intensities
        intensities = torch.clamp(intensities, min=self.eps)

        return intensities

    def loglike_loss(self, batch: tuple) -> tuple[torch.Tensor, int]:
        """
        Compute the log-likelihood loss for the Homogeneous Poisson process.
        log L = sum_k log(mu_{type_k}) - sum_i mu_i * T

        Args:
            batch: Tuple containing time_seq, time_delta_seq, type_seq, batch_non_pad_mask, _

        Returns:
            tuple: (total negative log-likelihood loss, number of events)
        """
        time_seq, _, type_seq, batch_non_pad_mask, _ = batch
        batch_size, seq_len = time_seq.shape
        device = self.device

        time_seq = time_seq.to(device)
        type_seq = type_seq.to(device)
        batch_non_pad_mask = batch_non_pad_mask.to(device)
        mu_device = self.mu.to(device) # Ensure mu is on the correct device

        # 1. Compute log intensity at each event time: sum_k log(mu_{type_k})
        # We only consider events k=1...L-1 that are not padding.
        # The first event (k=0) doesn't contribute to the sum term in standard formulations.
        valid_event_mask = batch_non_pad_mask[:, 1:] # [B, L-1], mask for events k=1 to L-1
        event_types_to_consider = type_seq[:, 1:] # [B, L-1]

        # Gather mu values for the types of events that occurred (k=1 to L-1)
        # mu_device: [D]
        # event_types_to_consider: [B, L-1]
        # Need to handle potential padding indices in event_types_to_consider if gather doesn't ignore them
        safe_event_types = event_types_to_consider.clone()
        safe_event_types[~valid_event_mask] = 0 # Replace pad id with 0 for gather index

        # mu_gathered: [B, L-1]
        mu_gathered = mu_device[safe_event_types]

        # Add epsilon for numerical stability before log
        log_mu_gathered = torch.log(mu_gathered + self.eps)

        # Apply mask to zero out log-likelihood for padded events
        masked_log_mu = log_mu_gathered * valid_event_mask

        # Sum over sequence length (k=1 to L-1)
        total_log_event_term = masked_log_mu.sum(dim=1) # [B]

        # 2. Compute the integral term: sum_i mu_i * T
        # T is the time of the last non-pad event in each sequence
        last_indices = batch_non_pad_mask.sum(dim=1) - 1
        # Handle sequences with zero length (all padding)
        last_indices = torch.clamp(last_indices, min=0)
        T = time_seq[torch.arange(batch_size, device=device), last_indices] # [B]

        # Sum of all base rates: sum_i mu_i
        total_mu = mu_device.sum() # Scalar

        # Total integral = (sum_i mu_i) * T
        total_integral = total_mu * T # [B]

        # 3. Compute total log-likelihood per sequence
        log_likelihood = total_log_event_term - total_integral # [B]

        # 4. Compute loss (negative log-likelihood sum over batch)
        loss = -log_likelihood.sum()

        # 5. Count number of valid events (excluding the first event and padding)
        num_events = valid_event_mask.sum().item()
        # Avoid division by zero if num_events is 0
        num_events = max(num_events, 1)

        return loss, num_events
