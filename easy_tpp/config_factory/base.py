"""
Base configuration classes and interfaces for the config factory.

This module provides the foundational classes and protocols that all
configuration classes should implement to ensure consistency and
type safety across the configuration system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, Union, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass, field
from copy import deepcopy

from easy_tpp.utils import Registrable

logger = logging.getLogger(__name__)

ConfigType = TypeVar('ConfigType', bound='BaseConfig')


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None):
        self.field_name = field_name
        super().__init__(message)


class ConfigSerializationError(Exception):
    """Raised when configuration serialization/deserialization fails."""
    pass


class ConfigBuilder(ABC):
    """
    Abstract base class for configuration builders.
    
    Provides a clean interface for building configurations from various sources
    while maintaining type safety and validation.
    """
    
    @abstractmethod
    def from_dict(self, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Build configuration from dictionary."""
        pass
    
    @abstractmethod
    def from_yaml_file(self, yaml_path: Union[str, Path]) -> 'BaseConfig':
        """Build configuration from YAML file."""
        pass
    
    @abstractmethod
    def validate(self, config: 'BaseConfig') -> bool:
        """Validate configuration."""
        pass


class ConfigValidator:
    """
    Configuration validator with extensible validation rules.
    
    Provides a framework for validating configuration objects with
    clear error reporting and customizable validation rules.
    """
    
    def __init__(self):
        self._validation_rules: List[callable] = []
    
    def add_rule(self, rule_func: callable) -> None:
        """Add a validation rule function."""
        self._validation_rules.append(rule_func)
    
    def validate(self, config: 'BaseConfig') -> List[str]:
        """
        Validate configuration and return list of error messages.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for rule in self._validation_rules:
            try:
                rule(config)
            except ConfigValidationError as e:
                error_msg = f"{e.field_name}: {str(e)}" if e.field_name else str(e)
                errors.append(error_msg)
            except Exception as e:
                errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def validate_required_fields(self, config: 'BaseConfig', required_fields: List[str]) -> None:
        """Validate that required fields are present and not None."""
        for field_name in required_fields:
            if not hasattr(config, field_name):
                raise ConfigValidationError(
                    f"Required field '{field_name}' is missing",
                    field_name=field_name
                )
            
            value = getattr(config, field_name)
            if value is None:
                raise ConfigValidationError(
                    f"Required field '{field_name}' cannot be None",
                    field_name=field_name
                )


@dataclass
class BaseConfig(Registrable, ABC):
    """
    Abstract base configuration class.
    
    Provides the foundation for all configuration classes with built-in
    validation, serialization, and type safety features.
    """
    
    def __post_init__(self):
        """Post-initialization hook for validation."""
        self.validate()
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required field names."""
        pass
    
    @abstractmethod
    def get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration as YAML-compatible dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls: Type[ConfigType], config_dict: Dict[str, Any]) -> ConfigType:
        """Create configuration from dictionary."""
        pass
    
    def validate(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        validator = ConfigValidator()
        validator.add_rule(lambda cfg: validator.validate_required_fields(cfg, self.get_required_fields()))
        
        errors = validator.validate(self)
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def copy(self: ConfigType) -> ConfigType:
        """
        Create a deep copy of the configuration.
        
        Returns:
            Deep copy of the configuration
        """
        return deepcopy(self)
    
    def update(self, **kwargs) -> None:
        """
        Update configuration fields.
        
        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Attempting to set unknown field '{key}' on {self.__class__.__name__}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.get_yaml_config()
    
    def save_to_yaml_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            file_path: Path to save the YAML file
        """
        from omegaconf import OmegaConf
        
        try:
            yaml_config = self.get_yaml_config()
            OmegaConf.save(yaml_config, file_path)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            raise ConfigSerializationError(f"Failed to save configuration to {file_path}: {str(e)}")
    
    @classmethod
    def load_from_yaml_file(cls: Type[ConfigType], file_path: Union[str, Path]) -> ConfigType:
        """
        Load configuration from YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Configuration instance
        """
        from omegaconf import OmegaConf
        
        try:
            config_dict = OmegaConf.load(file_path)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigSerializationError(f"Failed to load configuration from {file_path}: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"{self.__class__.__name__}({self.get_yaml_config()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__()


class ConfigFactory:
    """
    Factory class for creating configuration objects.
    
    Provides a centralized way to create and manage different types
    of configurations with built-in validation and error handling.
    """
    
    def __init__(self):
        self._builders: Dict[str, ConfigBuilder] = {}
        self._config_registry: Dict[str, Type[BaseConfig]] = {}
    
    def register_builder(self, config_type: str, builder: ConfigBuilder) -> None:
        """Register a configuration builder."""
        self._builders[config_type] = builder
        logger.debug(f"Registered builder for config type: {config_type}")
    
    def register_config_class(self, config_type: str, config_class: Type[BaseConfig]) -> None:
        """Register a configuration class."""
        self._config_registry[config_type] = config_class
        logger.debug(f"Registered config class: {config_class.__name__} for type: {config_type}")
    
    def create_config(self, config_type: str, config_dict: Dict[str, Any]) -> BaseConfig:
        """
        Create configuration from dictionary.
        
        Args:
            config_type: Type of configuration to create
            config_dict: Configuration data
            
        Returns:
            Configuration instance
            
        Raises:
            ValueError: If config type is not registered
            ConfigValidationError: If configuration is invalid
        """
        if config_type not in self._config_registry:
            available_types = list(self._config_registry.keys())
            raise ValueError(f"Unknown config type '{config_type}'. Available types: {available_types}")
        
        config_class = self._config_registry[config_type]
        
        try:
            return config_class.from_dict(config_dict)
        except Exception as e:
            raise ConfigValidationError(f"Failed to create {config_type} configuration: {str(e)}")
    
    def load_config_from_file(self, config_type: str, file_path: Union[str, Path]) -> BaseConfig:
        """
        Load configuration from file.
        
        Args:
            config_type: Type of configuration to load
            file_path: Path to configuration file
            
        Returns:
            Configuration instance
        """
        if config_type not in self._config_registry:
            available_types = list(self._config_registry.keys())
            raise ValueError(f"Unknown config type '{config_type}'. Available types: {available_types}")
        
        config_class = self._config_registry[config_type]
        return config_class.load_from_yaml_file(file_path)
    
    def get_available_config_types(self) -> List[str]:
        """Get list of available configuration types."""
        return list(self._config_registry.keys())


def config_class(config_type: str):
    """Decorator to register a config class with the config factory."""
    def decorator(cls):
        config_factory.register_config_class(config_type, cls)
        return cls
    return decorator


# Global factory instance
config_factory = ConfigFactory()