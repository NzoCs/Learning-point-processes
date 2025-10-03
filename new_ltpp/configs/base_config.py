"""
Base configuration classes and interfaces for the config factory.

This module provides the foundational classes and protocols that all
configuration classes should implement to ensure consistency and
type safety across the configuration system.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from typing_extensions import Self

from new_ltpp.utils import logger
from new_ltpp.configs.config_interface import (
    ConfigInterface,
    ConfigSerializationError, 
    ConfigValidationError
)

@dataclass
class Config(ABC):
    """
    Abstract base configuration class.

    Provides the foundation for all configuration classes with built-in
    validation, serialization, and type safety features.
    """

    def __post_init__(self):
        """Post-initialization hook for validation."""
        self.validate()

    def __str__(self):
        """String representation of the configuration."""
        

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required field names."""
        pass

    @abstractmethod
    def get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration as YAML-compatible dictionary."""
        pass

    def validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ConfigValidationError: If validation fails
        """
        # Import ici pour éviter les imports circulaires
        from new_ltpp.configs.config_utils import ConfigValidator
        
        validator = ConfigValidator()
        validator.add_rule(
            lambda cfg: validator.validate_required_fields(
                cfg, self.get_required_fields()
            )
        )

        errors = validator.validate(self)
        if errors:
            raise ConfigValidationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    def copy(self) -> Self:
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
                logger.warning(
                    f"Attempting to set unknown field '{key}' on {self.__class__.__name__}"
                )

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
            raise ConfigSerializationError(
                f"Failed to save configuration to {file_path}: {str(e)}"
            )
        
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"{self.__class__.__name__}({self.get_yaml_config()})"

    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__()