"""
Base configuration classes and interfaces for the config factory.

This module provides the foundational classes and protocols that all
configuration classes should implement to ensure consistency and
type safety across the configuration system.

- Config: ABC for runtime enforcement
- IConfig: Protocol for IDE type checking + isinstance()
"""

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Self, Union

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

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


class IConfig(Protocol):
    """Protocol for config - IDE type checking + isinstance() support."""

    def get_yaml_config(self) -> Dict[str, Any]: ...

    def update(self, **kwargs) -> None: ...

    def to_dict(self) -> Dict[str, Any]: ...

    def save_to_yaml_file(self, file_path: Union[str, Path]) -> None: ...


class Config(BaseModel):
    """
    Base configuration class backed by Pydantic.

    Provides the foundation for all configuration classes with built-in
    validation, serialization, and type safety features.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",
        protected_namespaces=(),
        frozen=True,
    )

    @classmethod
    def from_yaml(
        cls, yaml_path: Union[str, Path], path_in_yaml: Optional[str] = None
    ) -> Self:
        """
        Load configuration from a YAML file.
        If `path_in_yaml` is provided (e.g., 'training_configs.e500_b1'),
        it will extract that specific nested dictionary before instantiating.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                data = yaml.safe_load(f)

        if path_in_yaml:
            keys = path_in_yaml.split(".")
            for k in keys:
                if not isinstance(data, dict) or k not in data:
                    raise KeyError(f"Path '{path_in_yaml}' not found in {yaml_path}")
                data = data[k]

        try:
            return cls(**data)
        except ValidationError as e:
            raise ConfigValidationError(f"Pydantic Validation failed:\n{str(e)}")

    def get_yaml_config(self) -> Dict[str, Any]:
        """Return config as YAML-compatible dictionary."""
        return self.model_dump(mode="json")

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
        return self.model_dump(mode="json")

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
