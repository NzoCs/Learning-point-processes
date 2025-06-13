from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import torch
from easy_tpp.config_factory.base import BaseConfig, ConfigValidationError, config_factory, config_class
from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.config_factory.logger_config import LoggerConfig
from easy_tpp.utils import logger

@config_class('trainer_config')
@dataclass
class TrainerConfig(BaseConfig):
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
    """

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
    devices: Optional[int] = None
    save_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    save_model_dir: Optional[str] = None
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)
    dropout_rate: Optional[float] = None
    dropout: Optional[float] = None

    def __post_init__(self):
        # Directory setup
        ckpt = self.checkpoint_dir or "checkpoints"
        dirpath = self.save_dir or f"./{ckpt}/{self.model_id}/{self.dataset_id}/"
        os.makedirs(dirpath, exist_ok=True)
        self.save_model_dir = os.path.join(dirpath, "trained_models")
        os.makedirs(self.save_model_dir, exist_ok=True)
        # Logger config
        if not self.logger_config.save_dir:
            self.logger_config.save_dir = dirpath
        # Devices
        if self.devices is None:
            self.devices = self.detect_available_devices()
        # Dropout validation (optional)
        if self.dropout is not None:
            if not (0.0 <= self.dropout <= 1.0):
                raise ConfigValidationError("dropout must be between 0 and 1", "dropout")
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
        return {
            'dataset_id': self.dataset_id,
            'model_id': self.model_id,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'val_freq': self.val_freq,
            'patience': self.patience,
            'log_freq': self.log_freq,
            'checkpoints_freq': self.checkpoints_freq,
            'devices': self.devices,
            'save_model_dir': self.save_model_dir,
            'logger_config': self.logger_config.get_yaml_config() if hasattr(self.logger_config, 'get_yaml_config') and callable(self.logger_config.get_yaml_config) and getattr(self.logger_config, 'logger_type', None) is not None else {},
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'use_precision_16': self.use_precision_16
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainerConfig':
        logger_cfg = config_dict.get('logger_config', {})
        if not isinstance(logger_cfg, LoggerConfig):
            # Calculate save_dir early to pass to logger config
            ckpt = config_dict.get('checkpoint_dir', 'checkpoints')
            model_id = config_dict.get('model_id', 'unknown')
            dataset_id = config_dict.get('dataset_id', 'unknown')
            save_dir = config_dict.get('save_dir') or f"./{ckpt}/{model_id}/{dataset_id}/"
            logger_cfg = LoggerConfig.parse_from_yaml_config(logger_cfg, save_dir=save_dir)
        config_dict = dict(config_dict)
        config_dict['logger_config'] = logger_cfg
        # Alias: if 'dropout_rate' is present, also set 'dropout'
        if 'dropout_rate' in config_dict and 'dropout' not in config_dict:
            config_dict['dropout'] = config_dict['dropout_rate']
        return cls(**config_dict)

    def get_required_fields(self):
        return ['dataset_id', 'model_id']

@config_class('runner_config')
@dataclass
class RunnerConfig(BaseConfig):
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
            'trainer_config': self.trainer_config.get_yaml_config(),
            'model_config': self.model_config.get_yaml_config(),
            'data_config': self.data_config.get_yaml_config()
        }    
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RunnerConfig':
        """
        Create a RunnerConfig instance from a dictionary, ensuring all sub-configs are properly initialized.    
        """
        # Ensure all sub-configs are initialized correctly
        trainer = config_dict['trainer_config']
        if not isinstance(trainer, TrainerConfig):
            # Filter out invalid keys for TrainerConfig
            valid_keys = set(TrainerConfig.__dataclass_fields__.keys())
            filtered = {k: v for k, v in trainer.items() if k in valid_keys}
            dropped = set(trainer.keys()) - valid_keys
            if dropped:
                logger.warning(f"Filtered out invalid TrainerConfig keys: {dropped}")
            trainer = TrainerConfig.from_dict(filtered)
        model = config_dict['model_config']
        if not isinstance(model, ModelConfig):
            # Always convert sub-configs to dicts if possible
            if hasattr(model, 'get_yaml_config'):
                model = model.get_yaml_config()
            elif isinstance(model, dict):
                for subkey, subcls in [
                    ('base_config', getattr(ModelConfig, 'BaseConfig', None)),
                    ('specs', getattr(ModelConfig, 'ModelSpecsConfig', None)),
                    ('thinning', getattr(ModelConfig, 'ThinningConfig', None)),
                    ('simulation_config', getattr(ModelConfig, 'SimulationConfig', None)),
                ]:
                    if subkey in model and isinstance(model[subkey], dict) and subcls is not None:
                        model[subkey] = subcls.from_dict(model[subkey])
            # Filter out invalid keys for ModelConfig
            valid_keys = set(ModelConfig.__dataclass_fields__.keys())
            filtered = {k: v for k, v in model.items() if k in valid_keys}
            dropped = set(model.keys()) - valid_keys
            if dropped:
                logger.warning(f"Filtered out invalid ModelConfig keys: {dropped}")
            model = ModelConfig.from_dict(filtered)
        data = config_dict['data_config']
        if not isinstance(data, DataConfig):
            # Filter out invalid keys for DataConfig
            valid_keys = set(DataConfig.__dataclass_fields__.keys())
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            dropped = set(data.keys()) - valid_keys
            if dropped:
                logger.warning(f"Filtered out invalid DataConfig keys: {dropped}")
            data = DataConfig.from_dict(filtered)
        return cls(trainer_config=trainer, model_config=model, data_config=data)

    def get_required_fields(self):
        return ['trainer_config', 'model_config', 'data_config']