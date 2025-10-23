from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from new_ltpp.configs.config_factory import ConfigType, TrainingConfig

from .base_config_builder import ConfigBuilder


class TrainingConfigBuilder(ConfigBuilder):
    """
    Builder for TrainingConfig - configures training parameters and hyperparameters.

    This builder handles training settings like batch size, learning rate, epochs,
    validation frequency, early stopping, logging, and checkpointing.

    Required Parameters (must be set):
    ===================================
    - set_max_epochs(int): Maximum number of training epochs

    Optional Parameters:
    ===================
    Training Basics:
    - set_batch_size(int): Batch size for training (default: 32)
    - set_lr(float): Learning rate (default: 1e-3)
    - set_lr_scheduler(bool): Whether to use LR scheduler (default: True)
    - set_dropout(float): Dropout rate (default: 0.0)

    Validation & Early Stopping:
    - set_val_freq(int): Validation frequency in epochs (default: 10)
    - set_patience(int): Early stopping patience (default: 20)

    Logging & Checkpointing:
    - set_log_freq(int): Logging frequency in epochs (default: 5)
    - set_checkpoints_freq(int): Checkpoint saving frequency (default: 5)

    Advanced Training:
    - set_accumulate_grad_batches(int): Gradient accumulation batches (default: 1)
    - set_use_precision_16(bool): Use 16-bit precision (default: False)
    - set_devices(int): Number of devices to use (auto-detected if None)

    Usage Example:
    ==============

    .. code-block:: python

        builder = TrainingConfigBuilder()
        builder.set_max_epochs(100)
        builder.set_batch_size(64)
        builder.set_lr(1e-3)
        builder.set_lr_scheduler(True)
        builder.set_val_freq(5)
        builder.set_patience(10)
        training_config = builder.build()
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigType.TRAINING, config_dict)

    def from_dict(self, data: Dict[str, Any], training_config_path: str) -> List[str]:
        """Load training config from a nested dictionary path.

        Args:
            data: Dictionary containing configuration data
            training_config_path: Dotted path to training config (e.g., 'trainer_configs.quick_test')

        Returns:
            List of missing required fields
        """
        training_cfg = self._get_nested_value(data, training_config_path)
        self.config_dict = training_cfg
        return self.get_missing_fields()

    def load_from_yaml(
        self,
        yaml_path: Union[str, Path],
        training_config_path: str,
    ) -> List[str]:
        """Load training config from YAML file.

        Args:
            yaml_path: Path to YAML file
            training_config_path: Path to training config (e.g., 'trainer_configs.quick_test')

        Returns:
            List of missing required fields
        """
        data = self._load_yaml(yaml_path)
        return self.from_dict(data, training_config_path)

    def build(self, **kwargs) -> TrainingConfig:
        """Build and return a TrainingConfig instance."""
        return TrainingConfig(**self.get_config_dict())

    # TrainingConfig explicit parameters
    def set_max_epochs(self, max_epochs: int) -> List[str]:
        """Set maximum number of training epochs.

        Args:
            max_epochs: Maximum number of epochs
        """
        return self.set_field("max_epochs", max_epochs)

    def set_batch_size(self, batch_size: int) -> List[str]:
        """Set batch size for training.

        Args:
            batch_size: Number of samples per batch
        """
        return self.set_field("batch_size", batch_size)

    def set_lr(self, lr: float) -> List[str]:
        """Set learning rate.

        Args:
            lr: Learning rate value
        """
        return self.set_field("lr", lr)

    def set_lr_scheduler(self, lr_scheduler: bool) -> List[str]:
        """Set whether to use learning rate scheduler.

        Args:
            lr_scheduler: True to use LR scheduler
        """
        return self.set_field("lr_scheduler", lr_scheduler)

    def set_dropout(self, dropout: float) -> List[str]:
        """Set dropout rate.

        Args:
            dropout: Dropout rate (between 0 and 1)
        """
        return self.set_field("dropout", dropout)

    def set_val_freq(self, val_freq: int) -> List[str]:
        """Set validation frequency.

        Args:
            val_freq: Validation frequency in number of epochs
        """
        return self.set_field("val_freq", val_freq)

    def set_patience(self, patience: int) -> List[str]:
        """Set patience for early stopping.

        Args:
            patience: Number of epochs without improvement before stopping
        """
        return self.set_field("patience", patience)

    def set_log_freq(self, log_freq: int) -> List[str]:
        """Set logging frequency.

        Args:
            log_freq: Logging frequency in number of epochs
        """
        return self.set_field("log_freq", log_freq)

    def set_checkpoints_freq(self, checkpoints_freq: int) -> List[str]:
        """Set checkpoint saving frequency.

        Args:
            checkpoints_freq: Checkpoint frequency in number of epochs
        """
        return self.set_field("checkpoints_freq", checkpoints_freq)

    def set_accumulate_grad_batches(self, accumulate_grad_batches: int) -> List[str]:
        """Set number of batches for gradient accumulation.

        Args:
            accumulate_grad_batches: Number of batches to accumulate before weight update
        """
        return self.set_field("accumulate_grad_batches", accumulate_grad_batches)

    def set_use_precision_16(self, use_precision_16: bool) -> List[str]:
        """Set whether to use 16-bit precision.

        Args:
            use_precision_16: True to use 16-bit precision
        """
        return self.set_field("use_precision_16", use_precision_16)

    def set_devices(self, devices: int) -> List[str]:
        """Set number of devices to use.

        Args:
            devices: Number of devices (1 for single GPU, -1 for all available GPUs)
        """
        return self.set_field("devices", devices)

    def get_missing_fields(self) -> List[str]:
        """Return list of required fields that are missing."""
        required = ["max_epochs"]
        return [f for f in required if f not in self.config_dict]
