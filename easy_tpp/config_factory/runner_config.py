from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.config_factory.config import Config
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.config_factory.logger_config import LoggerConfig
from easy_tpp.utils import logger

import numpy as np
import torch
import os
import random

class TrainerConfig:
    
    @staticmethod
    def detect_available_devices():
        """Detect available hardware and return appropriate device settings."""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return "auto"  # Use CPU
    
    @staticmethod
    def generate_random_name():
        """Generate a random funny name for the experiment."""
        adjectives = ['happy', 'sleepy', 'grumpy', 'dancing', 'jumping', 'flying', 'mysterious']
        animals = ['panda', 'koala', 'penguin', 'octopus', 'unicorn', 'dragon', 'platypus']
        return f"{random.choice(adjectives)}_{random.choice(animals)}_{random.randint(1, 999)}"
    
    def __init__(self, **kwargs):
        
        
        self.stage = kwargs.get('stage', 'train')
        
        if self.stage not in ['train', 'test']:
            raise ValueError("Invalid stage. Choose either 'train' or 'test'.")
        
        if self.stage == 'test':
            kwargs['max_epochs'] = 1
            
        required_fields = ['max_epochs', 'dataset_id', "model_id"]
        # Check for required fields
        missing_fields = [field for field in required_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        self.dataset_id = kwargs.get('dataset_id')
        self.model_id = kwargs.get('model_id')
        
        # Setup save directories with random names
        save_dir = kwargs.get('save_dir', f"./{self.model_id}/{self.dataset_id}/")
        experiment_id = kwargs.get('experiment_id')
        
        if experiment_id is None:
            experiment_id = self.generate_random_name()
            
        dirpath = os.path.join(save_dir, experiment_id)
        os.makedirs(dirpath, exist_ok=True)
            
        logger_config = kwargs.get("logger_config", {})
        
        if logger_config.get("save_dir") is None:
            logger_config["save_dir"] = dirpath
            
        self.logger_config = LoggerConfig.parse_from_yaml_config(logger_config)
        
        self.dataset_id = kwargs.get('dataset_id', None)
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_epochs = kwargs.get('max_epochs')
        self.checkpoints_freq = kwargs.get('val_freq', 10)
        self.patience = kwargs.get('patience', np.inf)
        self.val_freq = kwargs.get('val_freq', 10)
        self.use_precision_16 = kwargs.get('use_precision_16', False)
        self.patience_max = kwargs.get("patience_max", float('inf'))
        
        self.log_freq = kwargs.get('log_freq', 1)
        self.accumulate_grad_batches = kwargs.get('accumulate_grad_batches', 1)
        # Auto-detect devices if not specified
        self.devices = kwargs.get('devices', self.detect_available_devices())
        self.save_model_dir = os.path.join(dirpath, "trained_models")
        os.makedirs(self.save_model_dir, exist_ok=True)
        
    def get(self, att):
        try:
            return getattr(self, att)
        except:
            return None
    
    def get_logger(self):
        
        return self.logger_config.configure_logger()
    
    @staticmethod
    def parse_from_yaml_config(yaml_config: dict) -> 'TrainerConfig':
        """Parse configuration from a YAML dictionary.
        
        Args:
            yaml_config (dict): Configuration dictionary from YAML
            **kwargs: Additional keyword arguments
            
        Returns:
            PLTrainerConfig: Configured trainer instance
            
        Raises:
            ValueError: If required parameters are missing
        """
            
        return TrainerConfig(**yaml_config)

    def get_yaml_config(self) -> dict:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: Configuration including all non-None parameters
        """
        config = {
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'val_freq': self.val_freq,
            'patience': self.patience,
            'log_freq': self.log_freq,
            'devices': self.devices,
            'save_model_dir': self.save_model_dir,
            'patience_max': self.patience_max if self.patience_max != float('inf') else None,
            "logger_config": self.logger_config.get_yaml_config(),
            "accumulate_grad_batches": self.accumulate_grad_batches
        }
        return {k: v for k, v in config.items() if v is not None}
        
    
@Config.register('runner_config')
class RunnerConfig(Config):
    """Configuration class for PyTorch Lightning Runner."""
    
    def __init__(self, trainer_config: TrainerConfig, model_config: ModelConfig, data_config: DataConfig):
        """Initialize PLRunnerConfig.
        
        Args:
            trainer_config (PLTrainerConfig): Training configuration
            model_config (ModelConfig): Model configuration
            data_config (PLDataConfig): Data configuration
        """
            
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.data_config = data_config

    @staticmethod
    def parse_from_yaml_config(yaml_config: dict, **kwargs) -> 'RunnerConfig':
        """Parse configuration from YAML dictionary.
        
        Args:
            yaml_config (dict): Configuration dictionary from YAML
            **kwargs: Additional keyword arguments
            
        Returns:
            PLRunnerConfig: Configured runner instance
            
        Raises:
            ValueError: If required configurations are missing
        """
        
        experiment_id = kwargs.get('experiment_id')
        dataset_id = kwargs.get('dataset_id')
        
        if experiment_id is not None : 
            exp_yaml_config = yaml_config[experiment_id]
        else : 
            exp_yaml_config = yaml_config
        
        #Initilize data_config
        data_loading_specs = exp_yaml_config.get('data_loading_specs', {})
        
        data_config_dict = yaml_config.get('data').get(dataset_id)
        data_config_dict['data_loading_specs'] = data_loading_specs

        data_config = DataConfig.parse_from_yaml_config(data_config_dict)
        
        #Initialize model_config and trainer_config
        model_config = exp_yaml_config.get('model_config', {})
        model_config['num_event_types'] = data_config.data_specs.num_event_types
        model_id = model_config.get('model_id', None)
        if model_id is None:
            raise ValueError("model_id is required in the model_config")
        
        trainer_config = exp_yaml_config.get('trainer_config')
        trainer_config['model_id'] = model_id
        trainer_config['dataset_id'] = dataset_id
        trainer_config = TrainerConfig.parse_from_yaml_config(trainer_config)
        
        model_config['base_config']['max_epochs'] = trainer_config.max_epochs
        
        model_config = ModelConfig.parse_from_yaml_config(model_config)
    
        return RunnerConfig(
            model_config=model_config,
            trainer_config=trainer_config,
            data_config=data_config
        )

    def get_yaml_config(self) -> dict:
        """Return the complete configuration in YAML compatible format.

        Returns:
            dict: Complete configuration including trainer, model and data configs
        """
        return {
            'trainer_config': self.trainer_config.get_yaml_config(),
            'model_config': self.model_config.get_yaml_config(),
            'data_config': self.data_config.get_yaml_config()
        }