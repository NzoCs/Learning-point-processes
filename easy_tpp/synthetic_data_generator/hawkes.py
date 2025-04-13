import torch
from torchtyping import TensorType

from easy_tpp.synthetic_data_generator.base_generator import BaseGenerator
from easy_tpp.config_factory.syn_gen_config import SynGenConfig


class Hawkes(BaseGenerator):
    """
    Implementation of the multivariate Hawkes process.
    
    A Hawkes process is a self-exciting point process where the occurrence of an event 
    increases the probability of future events. The intensity function of a Hawkes process
    is defined as:
        λ(t) = μ + Σ α * exp(-decay * (t - t_j))
    where:
        - μ is the baseline intensity
        - α represents the excitation parameters 
        - decay controls how quickly the excitation decays over time
    """
    
    def __init__(
        self,
        gen_config: SynGenConfig
    ):
        """
        Initialize the Hawkes process generator.
        
        Args:
            gen_config: Configuration object containing model parameters
                        with the following keys:
                        - "mu": Baseline intensity for each event type (list)
                        - "alpha": Excitation matrix between event types (list of lists)
                        - "decay": Decay rates for excitation (list of lists)
        """
        super(Hawkes, self).__init__(gen_config)
        
        model_config = gen_config.model_config
        self.baseline = torch.tensor(model_config.get("mu"), device=self.device)
        self.alpha = torch.tensor(model_config.get("alpha"), device=self.device)
        self.decay = torch.tensor(model_config.get("decay"), device=self.device)
            
    def compute_intensity_upper_bound(
        self,
        time_seq: TensorType["batch_size", "seq_len"],
        time_delta_seq: TensorType["batch_size", "seq_len"],
        event_seq: TensorType["batch_size", "seq_len"],
        compute_last_step_only: bool = False
    ) -> torch.Tensor:
        """
        Compute an upper bound on the total intensity of the process.
        
        This function estimates the total intensity of the model by summing the
        intensities of all event types at each time point in the provided sequence.
        
        Args:
            time_seq: Event occurrence times, shape [batch_size, seq_len]
            time_delta_seq: Time intervals between events, shape [batch_size, seq_len]
            event_seq: Event types associated with occurrence times, shape [batch_size, seq_len]
            compute_last_step_only: If True, compute intensity only for the last event
                                    in the sequence. Default: False
        
        Returns:
            torch.Tensor: Total intensity at each time point, shape depends on compute_last_step_only
        """
        # Calculate intensities for each event type at given times
        intensities = self.compute_intensities_at_sample_times(
            time_seq,
            time_delta_seq,
            event_seq,
            time_seq[:,:,None],  # Evaluate intensity at sequence times
            compute_last_step_only=compute_last_step_only
        ).squeeze(1)  # Remove unnecessary second dimension
        
        # Sum intensities over all event types to get total intensity
        total_intensity = intensities.sum(dim=-1)
        
        return total_intensity

    
        
    def compute_intensities_at_sample_times(
        self,
        time_seq: TensorType["batch_size", "seq_len"],
        time_delta_seq: TensorType["batch_size", "seq_len"],
        event_seq: TensorType["batch_size", "seq_len"],
        sampled_time_seq: TensorType["batch_size", "seq_len", "num_sample"],
        **kwargs
    ) -> TensorType["batch_size", "seq_len", "num_sample", "num_mark"]:
        """
        Compute the intensity of each event type at specified sampling times.
        
        Args:
            time_seq: Event occurrence times, shape [batch_size, seq_len]
            time_delta_seq: Time intervals between events, shape [batch_size, seq_len]
            event_seq: Event types associated with occurrence times, shape [batch_size, seq_len]
            sampled_time_seq: Times at which to evaluate intensities, shape [batch_size, seq_len, num_sample]
            **kwargs: Additional parameters:
                - compute_last_step_only: If True, compute intensity only for the last time step
        
        Returns:
            Tensor containing intensities for each event type at each sampled time,
            shape [batch_size, seq_len, num_sample, num_mark]
        """
        
        
        compute_last_step_only = kwargs.get("compute_last_step_only", False)
        
        batch_size, seq_len = time_seq.size()
        _, sample_len, num_sample = sampled_time_seq.size()
        num_mark = self.num_event_types
        
        # Process only the last step if requested
        if compute_last_step_only:
            sample_len = 1
            sampled_time_seq = sampled_time_seq[:, -1:, :]
        
        # Expand dimensions for broadcasting
        type_seq_expanded = event_seq.view(batch_size, 1, 1, 1, seq_len).expand(
            batch_size, sample_len, num_sample, num_mark, seq_len
        )
        time_seq_expanded = time_seq[:, None, None, None, :].repeat(
            1, sample_len, num_sample, num_mark, 1
        )
        
        # Create masks for valid time points and event types
        time_mask = torch.where(
            time_seq_expanded <= sampled_time_seq.view(batch_size, sample_len, num_sample, 1, 1),
            torch.tensor(1.0, device=self.device),
            torch.tensor(0.0, device=self.device)
        )
        
        mark_tensor = torch.arange(num_mark, device=self.device).float().view(1, 1, 1, num_mark, 1)
        mark_mask = torch.where(
            type_seq_expanded == mark_tensor,
            torch.tensor(1.0, device=self.device),
            torch.tensor(0.0, device=self.device)
        )
        
        mask = time_mask * mark_mask
        
        
        # Calculate time differences and exponential contributions
        time_contributions = sampled_time_seq.view(batch_size, sample_len, num_sample, 1, 1) - time_seq_expanded
        
        exp_contributions = self.alpha.view(num_mark, 1, 1, 1, num_mark, 1) * torch.exp(
            -self.decay.view(num_mark, 1, 1, 1, num_mark, 1) * 
            time_contributions.view(1, batch_size, sample_len, num_sample, num_mark, seq_len)
        )
        
        # Apply mask to include only valid contributions
        valid_exp_contributions = exp_contributions * mask.view(1, batch_size, sample_len, num_sample, num_mark, seq_len)
        
        # Compute final intensities: baseline + sum of excitations
        intensities = self.baseline.view(num_mark, 1, 1, 1) + valid_exp_contributions.sum(dim=-1).sum(dim=-1)
        intensities = intensities.permute(1, 2, 3, 0)
        
        return intensities