from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.config_factory.config import Config
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.config_factory.logger_config import LoggerConfig
from easy_tpp.utils import logger

import numpy as np
import torch
from datetime import datetime
import os

class TrainerConfig:
    
    @staticmethod
    def detect_available_devices():
        """Detect available hardware and return appropriate device settings."""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return "auto"  # Use CPU
    
    def __init__(self, **kwargs):
        
        required_fields = ['lr', 'max_epochs', 'save_dir']
        missing_fields = [field for field in required_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        
        if kwargs.get('save_dir') is None:
            datetime_format = "%Y-%m-%d_%H-%M-%S"
            date_time = datetime.now().strftime(datetime_format)
            model_id = kwargs.get('model_id', 'default_model')
            default_save_dir = f'./{model_id}/{kwargs.get('dataset_id')}/{date_time}'
            save_dir = default_save_dir
        else:
            save_dir = kwargs.get('save_dir')
            
        self.logger_type = kwargs.get("logger_type", "tensorboard")
        logger_save_dir = os.path.join(save_dir, "logs")
        logger_config = kwargs.get("logger_config", {})
        
        if logger_config.get("save_dir") is None:
            logger_config["save_dir"] = logger_save_dir
            
        self.logger_config = LoggerConfig(logger_config = logger_config, logger_type = self.logger_type)
        
        self.stage = kwargs.get('stage', 'train')
        self.dataset_id = kwargs.get('dataset_id', None)
        self.lr = kwargs.get('lr')
        self.lr_scheduler = kwargs.get('lr_scheduler')
        self.batch_size = kwargs.get('batch_size')
        self.max_epochs = kwargs.get('max_epochs')
        self.checkpoints_freq = kwargs.get('val_freq', 10)
        self.patience = kwargs.get('patience', np.inf)
        self.val_freq = kwargs.get('val_freq', 10)
        self.use_precision_16 = kwargs.get('use_precision_16', False)
        self.patience_max = kwargs.get("patience_max", float('inf'))
        
        self.log_freq = kwargs.get('log_freq', 1)
        # Auto-detect devices if not specified
        self.devices = kwargs.get('devices', self.detect_available_devices())
        self.save_model_dir = os.path.join(save_dir, "trained_models")
        
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
            
        return TrainerConfig(
            stage = yaml_config.get('stage', 'train'),
            dataset_id = yaml_config.get('dataset_id'),
            lr = yaml_config.get('lr'),
            use_precision_16 = yaml_config.get('use_precision_16', False),
            lr_scheduler = yaml_config.get('lr_scheduler', True),
            max_epochs = yaml_config.get('max_epochs'),
            val_freq = yaml_config.get('val_freq', 10),
            patience = yaml_config.get('patience', float('inf')),
            patience_max = yaml_config.get("patience_max", float('inf')),
            log_freq = yaml_config.get('log_freq', 1),
            # Auto-detect devices if not specified in the config
            devices = yaml_config.get('devices', TrainerConfig.detect_available_devices()),
            save_dir = yaml_config.get('save_dir'),
            logger_type = yaml_config.get('logger_type', 'tensorboard'),
            logger_config = yaml_config.get('logger_config'),
            model_id = yaml_config.get('model_id')
        )

    def get_yaml_config(self) -> dict:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: Configuration including all non-None parameters
        """
        config = {
            'lr': self.lr,
            'lr_scheduler': self.lr_scheduler,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'val_freq': self.val_freq,
            'patience': self.patience,
            'log_freq': self.log_freq,
            'devices': self.devices,
            'save_model_dir': self.save_model_dir,
            'patience_max': self.patience_max if self.patience_max != float('inf') else None,
            'logger_type': self.logger_type
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
        if not all([trainer_config, model_config, data_config]):
            raise ValueError("All configurations (trainer, model, data) must be provided")
            
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
        if experiment_id is not None : 
            exp_yaml_config = yaml_config[experiment_id]
        else : 
            exp_yaml_config = yaml_config
        
        #Initilize data_config
        data_config_dict = exp_yaml_config.get('data_config')
        
        if data_config_dict is None :
            
            try : 
                dataset_id = exp_yaml_config.get('trainer_config').get('dataset_id')
                data_config_dict = yaml_config.get('data').get(dataset_id)
                data_loading_specs = exp_yaml_config.get('data_loading_specs', {})
                data_config_dict["data_loading_specs"] = data_loading_specs
                
            except Exception as e:  # Changed from 'error' to catch all exceptions
                raise Exception(f"Error parsing data_config: {e}")

        data_config = DataConfig.parse_from_yaml_config(data_config_dict)
        
        #Initialize model_config
        model_config = exp_yaml_config.get('model_config')
        model_config['num_event_types'] = data_config.data_specs.num_event_types
        model_config = ModelConfig.parse_from_yaml_config(exp_yaml_config.get('model_config'))
        
        #initialize trainer_config
        trainer_config = exp_yaml_config.get('trainer_config')
        trainer_config['model_id'] = model_config.model_id
        trainer_config = TrainerConfig.parse_from_yaml_config(trainer_config)
        
    
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