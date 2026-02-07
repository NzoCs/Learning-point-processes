"""
Base configuration classes and interfaces for the config factory.

This module provides the foundational classes and protocols that all
configuration classes should implement to ensure consistency and
type safety across the configuration system.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, Protocol, runtime_checkable

from typing import Self
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


@dataclass
@runtime_checkable
class Config(Protocol):
    """
    Abstract base configuration class.

    Provides the foundation for all configuration classes with built-in
    validation, serialization, and type safety features.
    """

    @property
    def __name__(self) -> str:
        return self.__class__.__name__  # type: ignore

    def __init__(self): ...

    def __post_init__(self):
        self.validate()

    def get_required_fields(self) -> List[str]: ...

    def get_yaml_config(self) -> Dict[str, Any]: ...

    def validate(self) -> None:
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
