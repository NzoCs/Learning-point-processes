# new_ltpp/models/mixins/base_mixin.py
"""Base mixin class providing common initialization for mixins."""

from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl

from new_ltpp.shared_types import Batch
from new_ltpp.models.event_sampler import EventSampler


class BaseMixin(pl.LightningModule, ABC):
    """Base class for mixins providing shared interface.
    
    Note: This is a mixin and should not be instantiated directly.
    It declares attributes that concrete models must provide.
    """

    def __init__(
            self, 
            num_exp: int,
            device: torch.device,
            dtime_max: float,
            num_samples_boundary: int,
            over_sample_rate: float,
            ):
        
        """Initialize the BaseMixin.    

        Args:
            num_exp: Number of exponential random variables for thinning
            device: Device to run computations on
            dtime_max: Maximum time delta value in the dataset
            num_samples_boundary: Number of samples at boundary times
            over_sample_rate: Oversampling rate for thinning
        """
        
        super().__init__()

        self.num_exp = num_exp
        self._device = device
        self.dtime_max = dtime_max
        self.num_samples_boundary = num_samples_boundary
        self.over_sample_rate = over_sample_rate
        self._event_sampler = self.init_event_sampler()

    
    def init_event_sampler(self) -> EventSampler:
        """Get the event sampler used for thinning-based generation.

        Returns:
            EventSampler: The event sampler instance
        """
        return EventSampler(
            num_exp=self.num_exp,
            over_sample_rate=self.over_sample_rate,
            num_samples_boundary=self.num_samples_boundary,
            dtime_max=self.dtime_max,
            device=self._device,
        )
    
    @abstractmethod
    def compute_intensities_at_sample_times(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type."""
        raise NotImplementedError("Subclasses must implement compute_intensities_at_sample_times")

    @abstractmethod
    def loglike_loss(self, batch: Batch) -> tuple:
        """Compute the log-likelihood loss.
        
        Args:
            batch: Batch object containing sequences and masks
            
        Returns:
            Tuple of (loss, num_events)
        """
        raise NotImplementedError("Subclasses must implement loglike_loss")
    