"""
Base configuration classes and interfaces for the config factory.

This module provides the foundational classes and protocols that all
configuration classes should implement to ensure consistency and
type safety across the configuration system.

- Config: ABC for runtime enforcement
- IConfig: Protocol for IDE type checking + isinstance()
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Protocol, Self, Union, runtime_checkable

from new_ltpp.utils import logger


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, field_name: str = ""):
        super().__init__(message)
        self._field_name = field_name

    @property
    def field_name(self) -> str:
        return self._field_name


class ConfigSerializationError(Exception):
    """Raised when configuration serialization/deserialization fails."""

    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return f"ConfigSerializationError: {self.args[0]}"


@runtime_checkable
class IConfig(Protocol):
    """Protocol for config - IDE type checking + isinstance() support."""

    def get_required_fields(self) -> List[str]: ...

    def get_yaml_config(self) -> Dict[str, Any]: ...

    def validate(self) -> None: ...

    def copy(self) -> Self: ...

    def update(self, **kwargs) -> None: ...

    def to_dict(self) -> Dict[str, Any]: ...

    def save_to_yaml_file(self, file_path: Union[str, Path]) -> None: ...


class Config(ABC):
    """
    Abstract base configuration class - runtime enforcement via @abstractmethod.

    Provides the foundation for all configuration classes with built-in
    validation, serialization, and type safety features.
    """

    def __post_init__(self):
        self.validate()

    @classmethod
    @abstractmethod
    def get_required_fields(cls) -> List[str]:
        """Return list of required fields for this config."""
        pass

    @abstractmethod
    def get_yaml_config(self) -> Dict[str, Any]:
        """Return config as YAML-compatible dictionary."""
        pass

    def validate(self) -> None:
        """Validate the configuration."""
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
        """Create a deep copy of the configuration."""
        return deepcopy(self)

    def update(self, **kwargs) -> None:
        """Update configuration fields."""
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
        """Save configuration to YAML file."""
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
