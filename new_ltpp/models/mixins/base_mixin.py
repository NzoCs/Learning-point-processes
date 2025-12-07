# new_ltpp/models/mixins/base_mixin.py
"""Base mixin class providing common initialization for mixins."""

from abc import ABC, abstractmethod
from pathlib import Path

import pytorch_lightning as pl
import torch

from new_ltpp.globals import OUTPUT_DIR
from new_ltpp.models.event_sampler import EventSampler
from new_ltpp.shared_types import Batch


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
        pad_token_id: int,
        num_samples_boundary: int,
        over_sample_rate: float,
        output_dir: Path | str = OUTPUT_DIR,
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

        self.output_dir = Path(output_dir)
        self.num_exp = num_exp
        # Don't store device - use self.device property from PyTorch Lightning instead
        self.dtime_max = dtime_max
        self.num_samples_boundary = num_samples_boundary
        self.over_sample_rate = over_sample_rate
        # EventSampler will be initialized lazily to use correct device
        self._event_sampler = None
        self._init_device = device  # Only for initial module creation

        self.pad_token_id = pad_token_id

    def get_event_sampler(self) -> EventSampler:
        """Get or create the event sampler with the current device.
        
        This uses self.device (PyTorch Lightning property) which always
        reflects the current device of the model, not a cached value.

        Returns:
            EventSampler: The event sampler instance
        """
        if self._event_sampler is None or self._event_sampler.device != self.device:
            self._event_sampler = EventSampler(
                num_exp=self.num_exp,
                over_sample_rate=self.over_sample_rate,
                num_samples_boundary=self.num_samples_boundary,
                dtime_max=self.dtime_max,
                device=self.device,
            )
        return self._event_sampler

    @abstractmethod
    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_non_pad_mask: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        
    ) -> torch.Tensor:
        """Compute the intensity at sampled times, not only event times.

        Args:
            batch (Batch): batch input.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type."""
        pass

    @abstractmethod
    def loglike_loss(self, batch: Batch) -> tuple:
        """Compute the log-likelihood loss.

        Args:
            batch: Batch object containing sequences and masks

        Returns:
            Tuple of (loss, num_events)
        """
        raise NotImplementedError("Subclasses must implement loglike_loss")
