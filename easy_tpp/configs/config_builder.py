from abc import ABC, abstractmethod
from typing import Dict, Union, Any
from pathlib import Path

from easy_tpp.configs.base_config import Config

class ConfigBuilder(ABC):
    """
    Abstract base class for configuration builders.

    Provides a clean interface for building configurations from various sources
    while maintaining type safety and validation.
    """

    @abstractmethod
    def from_dict(self, config_dict: Dict[str, Any]) -> Config:
        """Build configuration from dictionary."""
        pass

    @abstractmethod
    def from_yaml_file(self, yaml_path: Union[str, Path]) -> Config:
        """Build configuration from YAML file."""
        pass

    @abstractmethod
    def validate(self, config: Config) -> bool:
        """Validate configuration."""
        pass
