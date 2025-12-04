"""Clean base model using mixins for separation of concerns."""

from abc import ABC, abstractmethod
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from new_ltpp.configs import ModelConfig
from new_ltpp.shared_types import Batch, DataInfo

from .mixins import TrainingMixin, VisualizationMixin
from .model_registry import RegistryMeta


class Model(
    TrainingMixin, VisualizationMixin, pl.LightningModule, ABC, metaclass=RegistryMeta
):
    """Base model class for all TPP models using mixins.

    Mixins provide:
    - TrainingMixin: training/validation/test/predict steps, prediction methods, simulation
    - VisualizationMixin: intensity graphs and plotting
    """

    def __init__(
        self, model_config: ModelConfig, data_info: DataInfo, output_dir: Path | str
    ):
        """Initialize the Model.

        Args:
            model_config: Model configuration
            data_info: Statistics from the dataset (num_event_types, end_time_max, dtime_max, pad_token_id)
        """

        # Simulation configuration for simulation mixin
        simulation_config = model_config.simulation_config

        self.seed = simulation_config.seed
        simulation_batch_size = simulation_config.batch_size
        simulation_start_time = data_info["end_time_max"]
        simulation_end_time = data_info["end_time_max"] + simulation_config.time_window
        initial_buffer_size = simulation_config.initial_buffer_size

        # Prediction configuration for prediction mixin, and event sampler
        thinning_config = model_config.thinning_config

        num_sample = thinning_config.num_sample
        num_samples_boundary = thinning_config.num_samples_boundary
        over_sample_rate = thinning_config.over_sample_rate
        num_exp = thinning_config.num_exp

        super().__init__(
            # BaseMixin params
            num_exp=num_exp,
            device=model_config.device,
            dtime_max=data_info["dtime_max"],
            num_samples_boundary=num_samples_boundary,
            over_sample_rate=over_sample_rate,
            output_dir=Path(output_dir),
            # SimulationMixin params
            simulation_start_time=simulation_start_time,
            simulation_end_time=simulation_end_time,
            num_event_types=data_info["num_event_types"],
            initial_buffer_size=initial_buffer_size,
            simulation_batch_size=simulation_batch_size,
            # PredictionMixin params
            num_sample=num_sample,
            num_step_gen=model_config.num_steps,
            # TrainingMixin paramok
            pad_token_id=data_info["pad_token_id"],
        )

        # Save hyperparameters
        self.save_hyperparameters()

        # Model configuration
        self.scheduler_config = model_config.scheduler_config
        self.eps = torch.finfo(torch.float32).eps

        # Loss computation configuration
        self.num_mc_samples = model_config.num_mc_samples
        self.use_mc_samples = model_config.use_mc_samples
        self.num_step_gen = model_config.num_steps

    def configure_optimizers(self):
        """Configure the optimizer for the model.

        Returns:
            optimizer: The optimizer to use for training.
        """

        # Use Adam optimizer with optional learning rate scheduler
        lr = self.scheduler_config.lr
        lr_scheduler = self.scheduler_config.lr_scheduler
        max_epochs = self.scheduler_config.max_epochs

        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Use cosine decay scheduler instead
        if hasattr(self, "lr_scheduler") and lr_scheduler:

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,  # Total number of epochs
                eta_min=lr * 0.01,  # Minimum learning rate at the end of schedule
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }

        return optimizer

    def make_dtime_loss_samples(self, time_delta_seq: torch.Tensor) -> torch.Tensor:
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """

        seq_len = time_delta_seq.size(1)

        # [1, 1, n_samples] - Monte Carlo sampling on [0,1]
        dtimes_ratio_sampled = torch.rand(
            1, seq_len, self.num_mc_samples, device=self.device
        )

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes

    @abstractmethod
    def loglike_loss(self, batch: Batch) -> tuple[torch.Tensor, int]:
        """Compute the log-likelihood loss for a batch of data.

        Args:
            batch: Batch containing time_seqs, time_delta_seqs, type_seqs, seq_non_pad_mask

        Returns:
            Tuple of (loss, number of events)
        """
        pass
