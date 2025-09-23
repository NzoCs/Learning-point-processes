import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from easy_tpp.configs.base_config import (
    Config,
    ConfigValidationError
)
from easy_tpp.configs.data_config import DataConfig
from easy_tpp.configs.logger_config import LoggerConfig
from easy_tpp.configs.model_config import ModelConfig
from easy_tpp.utils import logger


@dataclass
class TrainerConfig(Config):
    """
    Configuration for the Trainer, encapsulating training parameters and settings.
    Args:
        dataset_id (str): Identifier for the dataset.
        model_id (str): Identifier for the model.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        max_epochs (int, optional): Maximum number of training epochs. Defaults to None.
        val_freq (int, optional): Frequency of validation checks. Defaults to 10.
        patience (int, optional): Patience for early stopping. Defaults to 20.
        log_freq (int, optional): Frequency of logging. Defaults to 5.
        checkpoints_freq (int, optional): Frequency of saving checkpoints. Defaults to 5.
        accumulate_grad_batches (int, optional): Number of batches to accumulate gradients. Defaults to 1.
        use_precision_16 (bool, optional): Whether to use 16-bit precision. Defaults to False. (untested)
        activate_logging (bool, optional): Whether to activate logging. Defaults to False.
    """

    ROOT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
    dataset_id: str
    model_id: str
    batch_size: int = 32
    max_epochs: Optional[int] = None
    val_freq: int = 10
    patience: int = 20
    log_freq: int = 5
    checkpoints_freq: int = 5
    accumulate_grad_batches: int = 1
    use_precision_16: bool = False
    activate_logging: bool = True
    devices: Optional[int] = None
    save_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    logger_config: Optional[LoggerConfig] = None
    dropout: Optional[float] = None

    def __post_init__(self):
        # Directory setup
        ckpt = self.checkpoint_dir or "checkpoints"
        dirpath = self.ROOT_DIR / (
            self.save_dir or f"{ckpt}/{self.model_id}/{self.dataset_id}/"
        )
        self.save_model_dir = dirpath / "saved_model"
        self.save_model_dir.mkdir(parents=True, exist_ok=True)

        # Logger config - only if logging is activated
        if self.activate_logging:
            if self.logger_config is None:
                # Create default logger config if none provided
                self.logger_config = LoggerConfig(
                    save_dir=dirpath, logger_type="tensorboard"
                )
            elif not self.logger_config.save_dir:
                # Set save_dir if not already set
                self.logger_config.save_dir = dirpath

        # Devices
        if self.devices is None:
            self.devices = self.detect_available_devices()
        # Dropout validation (optional)
        if self.dropout is not None:
            if not (0.0 <= self.dropout <= 1.0):
                raise ConfigValidationError(
                    "dropout must be between 0 and 1", "dropout"
                )
        super().__post_init__()

    @staticmethod
    def detect_available_devices() -> int:
        try:
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return -1

    def get_yaml_config(self) -> Dict[str, Any]:
        config = {
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "val_freq": self.val_freq,
            "patience": self.patience,
            "log_freq": self.log_freq,
            "checkpoints_freq": self.checkpoints_freq,
            "devices": self.devices,
            "save_model_dir": self.save_model_dir,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "use_precision_16": self.use_precision_16,
            "activate_logging": self.activate_logging,
        }

        # Only include logger_config if logging is activated
        if self.activate_logging and self.logger_config is not None:
            config["logger_config"] = self.logger_config.get_yaml_config()

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainerConfig":
        from easy_tpp.configs.config_utils import ConfigValidator

        # 1. Validate the dictionary
        ConfigValidator.validate_required_fields(
            config_dict, cls._get_required_fields_list(), "TrainerConfig"
        )
        filtered_dict = ConfigValidator.filter_invalid_fields(config_dict, cls)

        # 2. Create sub-configuration instances if needed
        activate_logging = filtered_dict.get("activate_logging", False)
        if (
            activate_logging
            and "logger_config" in filtered_dict
            and isinstance(filtered_dict["logger_config"], dict)
        ):
            filtered_dict["logger_config"] = LoggerConfig.from_dict(
                filtered_dict["logger_config"]
            )
        elif not activate_logging:
            # Remove logger_config if logging is not activated
            filtered_dict.pop("logger_config", None)

        # 3. Create the instance
        return cls(**filtered_dict)

    @classmethod
    def _get_required_fields_list(cls) -> List[str]:
        """Get required fields as a list for validation."""
        return ["dataset_id", "model_id"]

    def get_required_fields(self):
        return ["dataset_id", "model_id"]


@dataclass
class RunnerConfig(Config):
    """
    Configuration for the Runner, encapsulating trainer, model, and data configurations.
    Args:
        trainer_config (TrainerConfig): Configuration for the training process.
        model_config (ModelConfig): Configuration for the model architecture and specifications.
        data_config (DataConfig): Configuration for the dataset and data processing.
    """

    trainer_config: TrainerConfig
    model_config: ModelConfig
    data_config: DataConfig

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "trainer_config": self.trainer_config.get_yaml_config(),
            "model_config": self.model_config.get_yaml_config(),
            "data_config": self.data_config.get_yaml_config(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Dict[str, Any]]) -> "RunnerConfig":
        from easy_tpp.configs.config_utils import ConfigValidator

        # 1. Validate the dictionary
        ConfigValidator.validate_required_fields(
            config_dict, cls._get_required_fields_list(), "RunnerConfig"
        )
        filtered_dict = ConfigValidator.filter_invalid_fields(config_dict, cls)

        # 2. Create sub-configuration instances
        trainer_config = filtered_dict["trainer_config"]
        if not isinstance(trainer_config, TrainerConfig):
            trainer_config = TrainerConfig.from_dict(trainer_config)

        model_config = filtered_dict["model_config"]
        if not isinstance(model_config, ModelConfig):
            model_config = ModelConfig.from_dict(model_config)

        data_config = filtered_dict["data_config"]
        if not isinstance(data_config, DataConfig):
            data_config = DataConfig.from_dict(data_config)

        # 3. Create the instance
        return cls(
            trainer_config=trainer_config,
            model_config=model_config,
            data_config=data_config,
        )

    @classmethod
    def _get_required_fields_list(cls) -> List[str]:
        """Get required fields as a list for validation."""
        return ["trainer_config", "model_config", "data_config"]

    def get_required_fields(self):
        return ["trainer_config", "model_config", "data_config"]
