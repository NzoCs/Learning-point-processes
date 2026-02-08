"""
Config Builder ABC + Protocol Pattern.
- ConfigBuilder: ABC for runtime enforcement
- IConfigBuilder: Protocol for IDE type checking + isinstance()
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, runtime_checkable

from new_ltpp.configs.config_factory import ConfigType, config_factory
from new_ltpp.configs.base_config import Config, IConfig


@runtime_checkable
class IConfigBuilder(Protocol):
    """Protocol for config builders - IDE type checking + isinstance() support."""

    @property
    def config_dict(self) -> Dict[str, Any]: ...

    @property
    def config_type(self) -> ConfigType: ...

    @property
    def required_fields(self) -> List[str]: ...

    def from_dict(self, config_dict: Dict[str, Any]) -> None: ...

    def build(self, **kwargs) -> Config: ...

    def get_clean_dict(self) -> Dict[str, Any]: ...

    @property
    def all_fields(self) -> List[str]: ...

    def get_unset_required_fields(self) -> List[str]: ...

    def get_unset_fields(self) -> List[str]: ...


class ConfigBuilder(ABC):
    """Abstract base class for config builders - runtime enforcement via @abstractmethod."""

    @property
    @abstractmethod
    def config_dict(self) -> Dict[str, Any]:
        """Return the current configuration dictionary."""
        pass

    @property
    @abstractmethod
    def config_type(self) -> ConfigType:
        """Return the config type for factory registration."""
        pass

    @property
    @abstractmethod
    def required_fields(self) -> List[str]:
        """Return list of required fields for this config."""
        pass

    @abstractmethod
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Initialize builder from an existing dictionary."""
        pass

    def build(self, **kwargs) -> IConfig:
        """Build a Config instance from the current dict via the factory."""
        return config_factory.create_config(
            self.config_type, self.config_dict, **kwargs
        )

    def get_clean_dict(self) -> Dict[str, Any]:
        """Return a clean dictionary without None values."""
        return {k: v for k, v in self.config_dict.items() if v is not None}

    @property
    def all_fields(self) -> List[str]:
        """Return all field names in the config dict."""
        return list(self.config_dict.keys())

    def get_unset_required_fields(self) -> List[str]:
        """Return required fields that are not set in the current config dict.

        Supports dotted paths for nested dicts (e.g. "data_loading_specs.batch_size").
        """
        unset: List[str] = []
        for f in self.required_fields:
            # support dotted nested keys
            if "." in f:
                parts = f.split(".")
                cur = self.config_dict
                missing = False
                for p in parts:
                    if not isinstance(cur, dict) or p not in cur or cur[p] is None:
                        missing = True
                        break
                    cur = cur[p]
                if missing:
                    unset.append(f)
            else:
                if self.config_dict.get(f) is None:
                    unset.append(f)
        return unset

    def get_unset_fields(self) -> List[str]:
        """Return all fields that are not set in the current config dict."""
        return [f for f in self.all_fields if self.config_dict.get(f) is None]
