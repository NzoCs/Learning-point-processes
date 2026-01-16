from typing import Any, Dict, Self, TypedDict, cast, List

from new_ltpp.configs.config_factory import TrainingConfig

from .base_config_builder import ConfigBuilder


class TrainingConfigDict(TypedDict):
    max_epochs: int | None
    batch_size: int | None
    lr: float | None
    lr_scheduler: bool | None
    dropout: float | None
    val_freq: int | None
    patience: int | None
    log_freq: int | None
    checkpoints_freq: int | None
    accumulate_grad_batches: int | None
    use_precision_16: bool | None
    devices: int | None
    save_dir: str | None


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
        (builder.set_max_epochs(100)
            .set_batch_size(64)
            .set_lr(1e-3)
            .set_lr_scheduler(True)
            .set_val_freq(5)
            .set_patience(10)
        )
        training_config = builder.build()
    """

    _config_dict: TrainingConfigDict

    def __init__(self):
        self._config_dict = {
            "max_epochs": None,
            "batch_size": None,
            "lr": None,
            "lr_scheduler": None,
            "dropout": None,
            "val_freq": None,
            "patience": None,
            "log_freq": None,
            "checkpoints_freq": None,
            "accumulate_grad_batches": None,
            "use_precision_16": None,
            "devices": None,
            "save_dir": None,
        }

    @property
    def config_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self._config_dict)
    
    @property
    def required_fields(self) -> List[str]:
        return ["max_epochs"]

    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load training config from a nested dictionary path.

        Args:
            config_dict: Dictionary containing configuration data
            training_config_path: Dotted path to training config (e.g., 'trainer_configs.quick_test')
        """
        self._config_dict = cast(TrainingConfigDict, config_dict)

    def build(self, **kwargs) -> TrainingConfig:
        """Build and return a TrainingConfig instance."""
        return TrainingConfig(**self.config_dict)

    # TrainingConfig explicit parameters
    def set_max_epochs(self, max_epochs: int) -> Self:
        """Set maximum number of training epochs.

        Args:
            max_epochs: Maximum number of epochs
        """
        self._config_dict["max_epochs"] = max_epochs
        return self

    def set_batch_size(self, batch_size: int) -> Self:
        """Set batch size for training.

        Args:
            batch_size: Number of samples per batch
        """
        self._config_dict["batch_size"] = batch_size
        return self

    def set_lr(self, lr: float) -> Self:
        """Set learning rate.

        Args:
            lr: Learning rate value
        """
        self._config_dict["lr"] = lr
        return self

    def set_lr_scheduler(self, lr_scheduler: bool) -> Self:
        """Set whether to use learning rate scheduler.

        Args:
            lr_scheduler: True to use LR scheduler
        """
        self._config_dict["lr_scheduler"] = lr_scheduler
        return self

    def set_dropout(self, dropout: float) -> Self:
        """Set dropout rate.

        Args:
            dropout: Dropout rate (between 0 and 1)
        """
        self._config_dict["dropout"] = dropout
        return self

    def set_val_freq(self, val_freq: int) -> Self:
        """Set validation frequency.

        Args:
            val_freq: Validation frequency in number of epochs
        """
        self._config_dict["val_freq"] = val_freq
        return self

    def set_patience(self, patience: int) -> Self:
        """Set patience for early stopping.

        Args:
            patience: Number of epochs without improvement before stopping
        """
        self._config_dict["patience"] = patience
        return self

    def set_log_freq(self, log_freq: int) -> Self:
        """Set logging frequency.

        Args:
            log_freq: Logging frequency in number of epochs
        """
        self._config_dict["log_freq"] = log_freq
        return self

    def set_checkpoints_freq(self, checkpoints_freq: int) -> Self:
        """Set checkpoint saving frequency.

        Args:
            checkpoints_freq: Checkpoint frequency in number of epochs
        """
        self._config_dict["checkpoints_freq"] = checkpoints_freq
        return self

    def set_accumulate_grad_batches(self, accumulate_grad_batches: int) -> Self:
        """Set number of batches for gradient accumulation.

        Args:
            accumulate_grad_batches: Number of batches to accumulate before weight update
        """
        self._config_dict["accumulate_grad_batches"] = accumulate_grad_batches
        return self

    def set_use_precision_16(self, use_precision_16: bool) -> Self:
        """Set whether to use 16-bit precision.

        Args:
            use_precision_16: True to use 16-bit precision
        """
        self._config_dict["use_precision_16"] = use_precision_16
        return self

    def set_devices(self, devices: int) -> Self:
        """Set number of devices to use.

        Args:
            devices: Number of devices (1 for single GPU, -1 for all available GPUs)
        """
        self._config_dict["devices"] = devices
        return self

    def set_save_dir(self, save_dir: str) -> Self:
        """Set directory to save training outputs.

        Args:
            save_dir: Path to save directory
        """
        self._config_dict["save_dir"] = save_dir
        return self
