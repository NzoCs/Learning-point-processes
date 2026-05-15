from enum import Enum
from typing import Any, Dict, Type, Union, Optional
from pathlib import Path

from new_ltpp.utils import logger
from .base_config import Config

from .data_config import DataConfig, DataLoadingSpecsConfig, TokenizerConfig
from .logger_config import LoggerConfig
from .model_config import (
    ModelConfig,
    ModelSpecsConfig,
    ThinningConfig,
)
from .runner_config import RunnerConfig, TrainingConfig, SimulationConfig


class ConfigType(Enum):
    """Registry of all available configuration types as (name, ConfigClass)."""

    TOKENIZER = ("tokenizer_config", TokenizerConfig)
    DATA_LOADING_SPECS = ("data_loading_specs_config", DataLoadingSpecsConfig)
    DATA = ("data_config", DataConfig)

    MODEL = ("model_config", ModelConfig)
    MODEL_SPECS = ("model_specs_config", ModelSpecsConfig)
    THINNING = ("thinning_config", ThinningConfig)
    SIMULATION = ("simulation_config", SimulationConfig)
    TRAINING = ("training_config", TrainingConfig)

    RUNNER = ("runner_config", RunnerConfig)
    LOGGER = ("logger_config", LoggerConfig)

    @property
    def config_name(self) -> str:
        return self.value[0]

    def get_class(self) -> Type[Config]:
        return self.value[1]

    def get_class_name(self) -> str:
        return self.get_class().__name__


class ConfigFactory:
    """Simple factory to create pydantic configuration instances."""

    def create_config(
        self, config_type: ConfigType, config_data: Dict[str, Any], **kwargs
    ) -> Config:
        config_class = config_type.get_class()
        logger.info(f"Creating configuration: {config_type.get_class_name()}")
        # With Pydantic, __init__ inherently validates. We just unpack everything.
        return config_class(**config_data, **kwargs)

    def create_config_by_name(
        self, config_name: str, config_data: Dict[str, Any], **kwargs
    ) -> Config:
        for config in ConfigType:
            if (
                config.config_name == config_name
                or config.get_class_name() == config_name
            ):
                return self.create_config(config, config_data, **kwargs)
        raise ValueError(f"Configuration '{config_name}' not found.")

    def create_from_yaml(
        self,
        config_type: ConfigType,
        yaml_path: Union[str, Path],
        path_in_yaml: Optional[str] = None,
    ) -> Config:
        """Create a configuration directly from a YAML file utilizing Pydantic's underlying from_yaml classmethod."""
        config_class = config_type.get_class()
        return config_class.from_yaml(yaml_path, path_in_yaml=path_in_yaml)


config_factory = ConfigFactory()
