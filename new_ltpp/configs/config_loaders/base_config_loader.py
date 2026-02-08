"""
Config Loader ABC + Protocol Pattern.
- ConfigLoader: ABC for runtime enforcement
- IConfigLoader: Protocol for IDE type checking + isinstance()
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Protocol, Union, runtime_checkable

import yaml


@runtime_checkable
class IConfigLoader(Protocol):
    """Protocol for config loaders - IDE type checking + isinstance() support."""

    def load_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML file with encoding fallback."""
        ...

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value via a dotted path (e.g., 'section.key')."""
        ...

    def load_from_dict(self, config_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Load configuration from an existing dictionary."""
        ...

    def load(self, yaml_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Load configuration and return a clean dictionary ready for a builder."""
        ...


class ConfigLoader(ABC):
    """Abstract base class for config loaders - runtime enforcement via @abstractmethod."""

    def load_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML file with encoding fallback."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                return yaml.safe_load(f)

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value via a dotted path (e.g., 'section.key')."""
        keys = path.split(".")
        current = data

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Path '{path}' not found in YAML")
            current = current[key]

        return current

    def load_from_dict(self, config_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Load configuration from an existing dictionary."""
        return config_dict

    @abstractmethod
    def load(self, yaml_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Load configuration and return a clean dictionary ready for a builder."""
        pass
