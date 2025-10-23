from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from new_ltpp.configs.config_factory import ConfigType
from new_ltpp.configs.runner_config import RunnerConfig

from .base_config_builder import ConfigBuilder
from .data_config_builder import DataConfigBuilder
from .model_config_builder import ModelConfigBuilder
from .training_config_builder import TrainingConfigBuilder


class RunnerConfigBuilder(ConfigBuilder):
    """
    Builder for RunnerConfig - orchestrates complete model training pipeline configuration.

    This builder manages three main sub-configurations:
    - DataConfig: Dataset and data loading settings
    - ModelConfig: Model architecture, simulation, and thinning settings
    - TrainingConfig: Training hyperparameters and optimization settings

    Required Parameters (must be set):
    ===================================
    Via sub-builders or set_* methods:
    - data_config: Complete data configuration (via data_builder or set_data_config)
    - model_config: Complete model configuration (via model_builder or set_model_config)
    - training_config: Training parameters (via training_builder or set_trainer_config)

    Essential Training Parameters:
    - training_builder.set_max_epochs(int): Number of training epochs
    - training_builder.set_batch_size(int): Batch size for training
    - training_builder.set_lr(float): Learning rate
    - model_builder.set_scheduler_config(bool, float): LR scheduler settings

    Optional Parameters:
    ===================
    - set_save_dir(str): Directory to save results (default: auto-generated)
    - set_logger_config(dict): Logging configuration (tensorboard, wandb, etc.)
    - training_builder.set_val_freq(int): Validation frequency in epochs
    - training_builder.set_patience(int): Early stopping patience
    - training_builder.set_dropout(float): Dropout rate
    - training_builder.set_devices(int): Number of devices to use
    - training_builder.set_accumulate_grad_batches(int): Gradient accumulation batches
    - training_builder.set_use_precision_16(bool): Use 16-bit precision

    Usage Examples:
    ===============

    .. code-block:: python

        # Method 1: Using sub-builders
        builder = RunnerConfigBuilder()
        builder.data_builder.set_dataset_id("my_dataset")
        builder.model_builder.set_specs({"hidden_size": 64})
        builder.training_builder.set_max_epochs(100)
        config = builder.build(model_id="NHP")

        # Method 2: Using pre-built configs
        builder = RunnerConfigBuilder()
        builder.set_data_config(my_data_config)
        builder.set_model_config(my_model_config)
        builder.set_trainer_config(my_training_config)
        config = builder.build(model_id="NHP")
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigType.RUNNER, config_dict)

        self.config_dict["model_config"] = {}
        self.config_dict["data_config"] = {}
        self.config_dict["training_config"] = {}

        self.model_builder = ModelConfigBuilder(self.config_dict["model_config"])
        self.data_builder = DataConfigBuilder(self.config_dict["data_config"])
        self.training_builder = TrainingConfigBuilder(
            self.config_dict["training_config"]
        )

    def from_dict(
        self,
        data: Dict[str, Any],
        training_config_path: str,
        model_config_path: str,
        data_config_path: str,
        data_loading_config_path: Optional[str] = None,
        data_specs_path: Optional[str] = None,
        simulation_config_path: Optional[str] = None,
        thinning_config_path: Optional[str] = None,
        logger_config_path: Optional[str] = None,
    ) -> List[str]:

        # Use TrainingConfigBuilder to load training config
        self.training_builder.from_dict(data, training_config_path)
        training_cfg = self.training_builder.get_config_dict()

        self.model_builder.from_dict(
            data, model_config_path, simulation_config_path, thinning_config_path
        )

        model_cfg = self.model_builder.get_config_dict()
        if "max_epochs" not in model_cfg["scheduler_config"]:
            model_cfg["scheduler_config"] = {
                "lr_scheduler": training_cfg.get("lr_scheduler"),
                "lr": training_cfg.get("lr"),
                "max_epochs": training_cfg.get("max_epochs"),
            }

        self.data_builder.from_dict(
            data, data_config_path, data_loading_config_path, data_specs_path
        )

        data_cfg = self.data_builder.get_config_dict()
        self.config_dict["training_config"] = training_cfg
        self.config_dict["model_config"] = model_cfg
        self.config_dict["data_config"] = data_cfg

        if logger_config_path:
            logger_cfg = self._get_nested_value(data, logger_config_path)
            self.config_dict["logger_config"] = logger_cfg

        return self.get_missing_fields()

    def build(self, *, model_id: str, **kwargs) -> RunnerConfig:
        # Ensure scheduler_config has max_epochs from training_config
        training_cfg = self.training_builder.get_config_dict()
        model_cfg = self.model_builder.get_config_dict()
        if "max_epochs" not in model_cfg.get("scheduler_config", {}):
            model_cfg["scheduler_config"] = {
                "lr_scheduler": training_cfg.get("lr_scheduler", True),
                "lr": training_cfg.get("lr", 1e-3),
                "max_epochs": training_cfg.get("max_epochs"),
            }
            self.config_dict["model_config"] = model_cfg

        return cast(RunnerConfig, super().build(model_id=model_id, **kwargs))

    def load_from_yaml(
        self,
        yaml_file_path: Union[str, Path],
        training_config_path: str,
        model_config_path: str,
        data_config_path: str,
        data_loading_config_path: Optional[str] = None,
        data_specs_path: Optional[str] = None,
        simulation_config_path: Optional[str] = None,
        thinning_config_path: Optional[str] = None,
        logger_config_path: Optional[str] = None,
    ) -> List[str]:
        """
        Load complete config from YAML using other builders.

        Args:
            yaml_file_path: Path to YAML file
            training_config_path: Path to training config (e.g., 'trainer_configs.quick_test')
            model_config_path: Path to model config (e.g., 'model_configs.neural_small')
            data_config_path: Path to data config (e.g., 'data_configs.test')
            data_loading_config_path: Path to data_loading_config (e.g., 'data_loading_configs.default')
            data_specs_path: Path to tokenizer_specs (e.g., 'tokenizer_specs.standard')
            simulation_config_path: Path to simulation config (e.g., 'simulation_configs.simulation_fast')
            thinning_config_path: Path to thinning config (e.g., 'thinning_configs.thinning_fast')
            logger_config_path: Path to logger config (e.g., 'logger_configs.csv')

        Returns:
            List of missing fields after loading
        """
        data = self._load_yaml(yaml_file_path)
        return self.from_dict(
            data,
            training_config_path,
            model_config_path,
            data_config_path,
            data_loading_config_path,
            data_specs_path,
            simulation_config_path,
            thinning_config_path,
            logger_config_path,
        )

    def set_trainer_config(self, trainer_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["training_config"] = trainer_cfg
        return self.get_missing_fields()

    def set_model_config(self, model_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["model_config"] = model_cfg
        return self.get_missing_fields()

    def set_data_config(self, data_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["data_config"] = data_cfg
        return self.get_missing_fields()

    # RunnerConfig explicit parameters
    def set_save_dir(self, save_dir: Union[str, Path]):
        """Set save directory for results.

        Args:
            save_dir: Path to save directory
        """
        return self.set_field("save_dir", save_dir)

    def set_logger_config(self, logger_cfg: Union[Dict[str, Any], Any]):
        """Set logger configuration.

        Args:
            logger_cfg: Logger configuration (tensorboard, wandb, csv, etc.)
        """
        self.config_dict["logger_config"] = logger_cfg
        return self.get_missing_fields()

    # TrainingConfig convenience methods
    def set_max_epochs(self, max_epochs: int):
        """Set maximum number of training epochs in training_config.

        Args:
            max_epochs: Maximum number of epochs
        """
        return self.training_builder.set_max_epochs(max_epochs)

    def set_batch_size(self, batch_size: int):
        """Set batch size in training_config.

        Args:
            batch_size: Number of samples per batch
        """
        return self.training_builder.set_batch_size(batch_size)

    def set_lr(self, lr: float):
        """Set learning rate in training_config.

        Args:
            lr: Learning rate value
        """
        return self.training_builder.set_lr(lr)

    def set_lr_scheduler(self, lr_scheduler: bool):
        """Set whether to use learning rate scheduler in training_config.

        Args:
            lr_scheduler: True to use LR scheduler
        """
        return self.training_builder.set_lr_scheduler(lr_scheduler)

    def set_dropout(self, dropout: float):
        """Set dropout rate in training_config.

        Args:
            dropout: Dropout rate (between 0 and 1)
        """
        return self.training_builder.set_dropout(dropout)

    def set_val_freq(self, val_freq: int):
        """Set validation frequency in training_config.

        Args:
            val_freq: Validation frequency in number of epochs
        """
        return self.training_builder.set_val_freq(val_freq)

    def set_patience(self, patience: int):
        """Set patience for early stopping in training_config.

        Args:
            patience: Number of epochs without improvement before stopping
        """
        return self.training_builder.set_patience(patience)

    def set_log_freq(self, log_freq: int):
        """Set logging frequency in training_config.

        Args:
            log_freq: Logging frequency in number of epochs
        """
        return self.training_builder.set_log_freq(log_freq)

    def set_checkpoints_freq(self, checkpoints_freq: int):
        """Set checkpoint saving frequency in training_config.

        Args:
            checkpoints_freq: Checkpoint frequency in number of epochs
        """
        return self.training_builder.set_checkpoints_freq(checkpoints_freq)

    def set_accumulate_grad_batches(self, accumulate_grad_batches: int):
        """Set number of batches for gradient accumulation in training_config.

        Args:
            accumulate_grad_batches: Number of batches to accumulate before weight update
        """
        return self.training_builder.set_accumulate_grad_batches(
            accumulate_grad_batches
        )

    def set_use_precision_16(self, use_precision_16: bool):
        """Set whether to use 16-bit precision in training_config.

        Args:
            use_precision_16: True to use 16-bit precision
        """
        return self.training_builder.set_use_precision_16(use_precision_16)

    def set_scheduler_config(
        self, lr_scheduler: bool, lr: float, max_epochs: Optional[int] = None
    ):
        """Set learning rate scheduler configuration for the model.

        Args:
            lr_scheduler: Whether to use learning rate scheduler
            lr: Learning rate
            max_epochs: Maximum number of training epochs. If not provided, will be taken from training_config.
        """
        if max_epochs is None:
            # Try to get max_epochs from training_config
            training_config = self.training_builder.get_config_dict()
            max_epochs = training_config.get("max_epochs")
            if max_epochs is None:
                raise ValueError(
                    "max_epochs must be provided either as argument to set_scheduler_config() "
                    "or by calling training_builder.set_max_epochs() first."
                )

        return self.model_builder.set_scheduler_config(lr_scheduler, lr, max_epochs)

    def get_missing_fields(self) -> List[str]:
        required = ["training_config", "model_config", "data_config"]
        return [f for f in required if f not in self.config_dict]
