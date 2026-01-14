from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from new_ltpp.configs.base_config import Config
from new_ltpp.configs.config_factory import ConfigType, config_factory


class ConfigBuilder(ABC):
    """Interface for a specific config builder."""

    def __init__(
        self, config_type: ConfigType, config_dict: Optional[Dict[str, Any]] = None
    ):
        self.config_type = config_type
        # Use 'is None' instead of 'or' to preserve empty dict references
        self.config_dict = config_dict if config_dict is not None else {}

    def set_field(self, field: str, value: Any):
        self.config_dict[field] = value
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        # Default no constraints; subclasses should override
        return []

    def get_config_dict(self) -> Dict[str, Any]:
        return self.config_dict

    def build(self, **kwargs) -> Config:
        """
        Build a Config instance from the current dict via the factory.
        Args:
            **kwargs: passed to factory.create_config/create_config_by_name
        """

        return config_factory.create_config(
            self.config_type, self.get_config_dict(), **kwargs
        )

    def _load_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
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

    @abstractmethod
    def load_from_yaml(self, yaml_path: Union[str, Path], *args, **kwargs) -> List[str]:
        """Load a YAML file with encoding fallback."""
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any], *args, **kwargs) -> List[str]:
        pass

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value via a dotted path (e.g., 'section.key')."""
        keys = path.split(".")
        current = data

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Path '{path}' not found in YAML")
            current = current[key]

        return current
