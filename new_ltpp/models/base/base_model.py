# new_ltpp/models/mixins/base_mixin.py
"""Base mixin providing the minimal shared interface for all TPP model mixins."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim

from new_ltpp.globals import OUTPUT_DIR
from new_ltpp.models.event_sampler import EventSampler
from new_ltpp.shared_types import Batch

# Tu devras t'assurer que ces imports correspondent bien à ton architecture
from new_ltpp.models.model_registry import RegistryMeta
from new_ltpp.configs import ModelConfig
from new_ltpp.shared_types import DataInfo


class BaseModel(
    pl.LightningModule, ABC, metaclass=RegistryMeta
):  # Ajoute `metaclass=RegistryMeta` si nécessaire
    """Base model class for all TPP models.

    Responsabilités fusionnées :
        - Interface principale de modèle (loglike_loss, compute_intensities)
        - Configuration des optimiseurs (configure_optimizers)
        - Échantillonnage de Monte-Carlo (make_dtime_loss_samples)
        - Initialisation et délégation au Simulator
        - Instanciation paresseuse (device-aware) de l'EventSampler
    """

    def __init__(
        self,
        model_config: ModelConfig,
        data_info: DataInfo,
        output_dir: Union[Path, str] = OUTPUT_DIR,
        **kwargs,
    ):
        """Initialize the BaseModel.

        Args:
            model_config: Full model configuration.
            data_info: Dataset statistics (num_event_types, end_time_max, dtime_max, pad_token_id).
            output_dir: Directory for saving artefacts.
            simulation_config: Simulation config (owned by RunnerConfig, not ModelConfig).
            statistical_test_config: Configuration for statistical tests.
        """
        super().__init__()  # Initialize LightningModule
        self.save_hyperparameters()

        # --- Base Data & Paths ---
        self.output_dir = Path(output_dir)
        self.num_event_types: int = data_info["num_event_types"]
        self.dtime_max: float = data_info["dtime_max"]
        self.pad_token_id: int = data_info["pad_token_id"]

        self.scheduler_config = model_config.scheduler_config
        self.eps = torch.finfo(torch.float32).eps

        # --- Thinning / Prediction config ---
        thinning_config = model_config.thinning_config
        self.num_exp: int = thinning_config.num_exp
        self.num_samples_boundary: int = thinning_config.num_samples_boundary
        self.over_sample_rate: float = thinning_config.over_sample_rate
        self.num_mc_samples: int = thinning_config.num_sample

        # EventSampler est créé de manière paresseuse pour garantir le bon device (géré par Lightning)
        self._event_sampler: Optional[EventSampler] = None
        self._init_device = model_config.device

    # ------------------------------------------------------------------
    # EventSampler — lazy, device-aware
    # ------------------------------------------------------------------

    def get_event_sampler(self) -> EventSampler:
        """Return (or lazily create) the EventSampler on the current device."""
        if self._event_sampler is None or self._event_sampler.device != self.device:
            self._event_sampler = EventSampler(
                num_exp=self.num_exp,
                over_sample_rate=self.over_sample_rate,
                num_samples_boundary=self.num_samples_boundary,
                dtime_max=self.dtime_max,
                device=self.device,
            )
        return self._event_sampler

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute intensity values at arbitrary sampled times.

        Args:
            time_seqs: Cumulative event times [batch_size, seq_len].
            time_delta_seqs: Inter-event times [batch_size, seq_len].
            type_seqs: Event type sequences [batch_size, seq_len].
            sample_dtimes: Time deltas to evaluate [batch_size, seq_len, num_samples].
            compute_last_step_only: If True, only compute for the last position.

        Returns:
            Intensity values [batch_size, seq_len, num_samples, num_event_types].
        """
        ...

    @abstractmethod
    def loglike_loss(self, batch: Batch) -> tuple:
        """Compute the log-likelihood loss.

        Args:
            batch: Batch object with sequences and masks.

        Returns:
            Tuple of (loss, num_events).
        """
        ...

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Union[dict[str, Any], optim.Optimizer]:  # type: ignore[override]
        lr = self.scheduler_config.lr
        use_scheduler = self.scheduler_config.lr_scheduler
        max_epochs = self.scheduler_config.max_epochs

        optimizer = optim.Adam(self.parameters(), lr=lr)

        if use_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=lr * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }

        return optimizer

    # ------------------------------------------------------------------
    # Monte-Carlo sampling helper
    # ------------------------------------------------------------------

    def make_dtime_loss_samples(self, time_delta_seqs: torch.Tensor) -> torch.Tensor:
        """Generate uniform MC samples inside each inter-event interval.

        Args:
            time_delta_seqs: [batch_size, seq_len]

        Returns:
            [batch_size, seq_len, num_mc_samples]
        """
        seq_len = time_delta_seqs.size(1)
        dtimes_ratio_sampled = torch.rand(
            1, seq_len, self.num_mc_samples, device=self.device
        )
        return time_delta_seqs[:, :, None] * dtimes_ratio_sampled


class NeuralModel(BaseModel, ABC):
    """Neural Temporal Point Process model base class.

    Adds hidden_size, dropout, and the type embedding shared by all neural models.
    """

    def __init__(
        self,
        *,
        dropout: float,
        hidden_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.layer_type_emb = nn.Embedding(
            num_embeddings=self.num_event_types + 1,
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_token_id,
            device=self.device,
        )
