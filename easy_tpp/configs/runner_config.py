import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from easy_tpp.configs.base_config import (
    Config,
    ConfigValidationError
)
from easy_tpp.configs.data_config import DataConfig
from easy_tpp.configs.logger_config import LoggerConfig
from easy_tpp.configs.model_config import ModelConfig



@dataclass
class TrainingConfig(Config):
    """
    Configuration for the Training, encapsulating training parameters and settings.
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

    max_epochs: int
    batch_size: int = 32
    lr : float = 1e-3
    lr_scheduler : bool = True
    dropout: float = 0.0
    val_freq: int = 10
    patience: int = 20
    log_freq: int = 5
    checkpoints_freq: int = 5
    accumulate_grad_batches: int = 1
    use_precision_16: bool = False
    devices: Optional[int] = None

    def __post_init__(self):

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
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "val_freq": self.val_freq,
            "patience": self.patience,
            "log_freq": self.log_freq,
            "checkpoints_freq": self.checkpoints_freq,
            "devices": self.devices,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "use_precision_16": self.use_precision_16
        }

        return config

    def get_required_fields(self):
        return ["max_epochs"]


@dataclass
class RunnerConfig(Config):
    """
    Configuration for the Runner, encapsulating trainer, model, and data configurations.
    Args:
        training_config (dict): Dictionnaire de configuration pour le training.
        model_config (dict): Dictionnaire de configuration pour le modèle.
        data_config (dict): Dictionnaire de configuration pour les données.
    """

    ROOT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


    training_config: TrainingConfig
    model_config: ModelConfig
    data_config: DataConfig
    model_id: str
    logger_config: Optional[LoggerConfig] = None
    activate_logging: bool = False
    save_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None

    def __init__(
            self, 
            model_id: str, 
            training_config: Union[TrainingConfig, dict], 
            model_config: Union[ModelConfig, dict], 
            data_config: Union[DataConfig, dict],
            logger_config: Optional[Union[LoggerConfig, dict]] = None,
            activate_logging: bool = False,
            save_dir: Optional[str] = None,
            checkpoint_dir: Optional[str] = None,
            **kwargs
            ):

        # assign simple attributes first so they are available during setup
        self.logger_config = logger_config
        self.activate_logging = activate_logging
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir

        # Instancie les configs intermédiaires à partir des dicts
        self.training_config = training_config if isinstance(training_config, TrainingConfig) else TrainingConfig(**training_config)
        self.model_config = model_config if isinstance(model_config, ModelConfig) else ModelConfig(**model_config)
        self.data_config = data_config if isinstance(data_config, DataConfig) else DataConfig(**data_config)

        # dataset id from data config
        self.dataset_id = self.data_config.dataset_id
        self.model_id = model_id

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
            elif not getattr(self.logger_config, "save_dir", None):
                # Set save_dir if not already set
                self.logger_config.save_dir = dirpath

        super().__init__(**kwargs)

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "training_config": self.training_config.get_yaml_config(),
            "model_config": self.model_config.get_yaml_config(),
            "data_config": self.data_config.get_yaml_config(),
        }

    def get_required_fields(self):
        return ["training_config", "model_config", "data_config"]