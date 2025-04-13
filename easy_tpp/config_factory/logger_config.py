from enum import Enum
from typing import Dict, Any, Optional, Union, List, Type, get_type_hints
from abc import ABC, abstractmethod
import os
import importlib
from contextlib import contextmanager
import copy
from pytorch_lightning.loggers import (
    WandbLogger,
    TensorBoardLogger,
    CSVLogger,
    MLFlowLogger,
    NeptuneLogger,
    CometLogger  # for example, for Comet.ml
)
from easy_tpp.config_factory.config import Config
from easy_tpp.utils import logger

class LoggerType(Enum):
    CSV = 'csv'
    WandB = 'wandb'
    MLFLOW = 'mlflow'
    COMET = 'comet'
    NEPTUNE = 'neptune'
    TENSORBOARD = 'tensorboard'

class BaseLoggerAdapter(ABC):
    """Abstract class defining the interface for all logger adapters, see the documentation of different loggers to   
    understand the various parameters."""
    
    @classmethod
    def get_required_params(cls) -> List[str]:
        """Returns the list of required parameters for this logger"""
        return []
    
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and completes the configuration"""
        
        # Check required parameters
        for param in cls.get_required_params():
            if param not in config:
                raise ValueError(f"Parameter '{param}' is required for {cls.__name__}")
            
        return config
    
    @classmethod
    @abstractmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        """Configures and returns a logger instance"""
        pass
    
    
class CSVLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        """
        Returns the list of required parameters for CSVLogger.
        """
        return ['save_dir', 'name']

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        """
        Configures and returns an instance of CSVLogger using the provided parameters.
        """
        # Create the save directory if necessary
        os.makedirs(config['save_dir'], exist_ok=True)
        
        return CSVLogger(**config)

        


class WandBLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        """
        Returns the list of required parameters for WandbLogger.
        """
        return ["project"]  # 'project' is the only strictly required parameter
    
    
    @classmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        """
        Configures and returns an instance of WandbLogger using the provided parameters.
        """
        
        save_dir = config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        try :
            import wandb
            wandb.finish()
            wandb.init(dir = save_dir)
            
            return WandbLogger(**config)
        
        except :
            return WandbLogger(**config)
    
    
class MLflowLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        """
        Returns the list of required parameters for MLFlowLogger.
        """
        return ['experiment_name']

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        """
        Checks required parameters, configures and returns an instance of MLFlowLogger
        using the provided parameters.
        """
        return MLFlowLogger(**config)



class CometLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ['api_key', 'project_name']
    
    @classmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        
        return CometLogger(**config)


class NeptuneLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ['api_token', 'project']
    
    @classmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        
        return NeptuneLogger(**config)

class TensorboardLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ['save_dir', 'name']
    
    @classmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        
        
        return TensorBoardLogger(**config)

# Registry of adapters
LOGGER_ADAPTERS = {
    LoggerType.CSV: CSVLoggerAdapter,
    LoggerType.WandB: WandBLoggerAdapter,
    LoggerType.MLFLOW: MLflowLoggerAdapter,
    LoggerType.COMET: CometLoggerAdapter,
    LoggerType.NEPTUNE: NeptuneLoggerAdapter,
    LoggerType.TENSORBOARD: TensorboardLoggerAdapter
}

class  LoggerConfig(Config):
    """
    Class to configure a specific logger type.
    
    This class allows configuring a single logger at a time
    based on the specified LoggerType.
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the configuration for a specific logger type.
        
        Args:
            logger_type: Type of logger to configure (LoggerType or string)
            **kwargs: Configuration parameters specific to the logger type
        
        Raises:
            ValueError: If the logger type is unknown or required parameters are missing
            TypeError: If the logger type is not a LoggerType or str
        """
        
        logger_type = kwargs.get('logger_type', None)
        
        # Convert string to LoggerType if necessary
        if isinstance(logger_type, str):
            try:
                self.logger_type = LoggerType(logger_type)
            except ValueError:
                raise ValueError(f"Unknown logger type: {logger_type}")
        else:
            self.logger_type = logger_type
        
        # Get the adapter for this logger type
        self.adapter = LOGGER_ADAPTERS.get(self.logger_type)
        if not self.adapter:
            logger.warning(f"No adapter available for logger type: {self.logger_type}")
            logger.warning("The logger will not be configured.")
        
        # Validate and initialize the configuration
        if self.adapter is not None :
            self.config = self.adapter.validate_config(kwargs.get("logger_config", {}))
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retrieves the current configuration, possibly modified by the mode.
        
        Returns:
            Dict[str, Any]: Logger configuration
        """
        config = self.config.copy()
        return config
    
    def get_yaml_config(self):
        """Get the yaml format config from self.

        Returns:
            dict: Configuration in yaml format
        """
        yaml_config = {
            'logger_type': self.logger_type.value,
            'logger_config': self.config
        }
        return yaml_config

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            LoggerConfig: Config class for logger.
        """
        logger_type = yaml_config.get('logger_type')
        config = yaml_config.get('config', {})
        
        # Update config with any additional kwargs
        if kwargs:
            config.update(kwargs)
            
        return LoggerConfig(logger_type, **config)

    def copy(self):
        """Get a same and freely modifiable copy of self.

        Returns:
            LoggerConfig: A copy of the current config.
        """
        return LoggerConfig(self.logger_type, **copy.deepcopy(self.config))
    
    def configure_logger(self) -> Any:
        """
        Configures and returns a logger instance according to the specified type.
            
        Returns:
            Any: Configured logger instance
            
        Raises:
            ImportError: If the required dependencies are not installed
        """
        config = self.get_config()
        return self.adapter.configure(config)
    
    @classmethod
    def list_required_params(cls, logger_type: Union[LoggerType, str]) -> List[str]:
        """
        Lists required parameters for a logger type.
        
        Args:
            logger_type: Logger type
            
        Returns:
            List[str]: List of required parameters
        """
        if isinstance(logger_type, str):
            try:
                logger_type = LoggerType(logger_type)
            except ValueError:
                raise ValueError(f"Unknown logger type: {logger_type}")
                
        adapter = LOGGER_ADAPTERS.get(logger_type)
        if not adapter:
            raise ValueError(f"No adapter available for logger type: {logger_type}")
            
        return adapter.get_required_params()