"""
Simple factory for TPP configurations

This factory makes it easy to instantiate configuration objects using
an enum of the available configuration types.

Usage:
    from new_ltpp.configs.config_factory import ConfigType, ConfigFactory

    factory = ConfigFactory()
    config = factory.create_config(ConfigType.MODEL, config_data)
"""

from enum import Enum
from typing import Any, Type

from new_ltpp.utils import logger

from .base_config import Config, ConfigValidationError
from .data_config import DataConfig, DataLoadingSpecsConfig, TokenizerConfig
from .hpo_config import HPOConfig, HPORunnerConfig
from .logger_config import LoggerConfig
from .model_config import (
    ModelConfig,
    ModelSpecsConfig,
    SimulationConfig,
    ThinningConfig,
)
from .runner_config import RunnerConfig, TrainingConfig


class ConfigType(Enum):
    """Registry of all available configuration types as (name, ConfigClass)."""

    # Data configurations
    TOKENIZER = ("tokenizer_config", TokenizerConfig)
    DATA_LOADING_SPECS = ("data_loading_specs_config", DataLoadingSpecsConfig)
    DATA = ("data_config", DataConfig)

    # Model configurations
    MODEL = ("model_config", ModelConfig)
    MODEL_SPECS = ("model_specs_config", ModelSpecsConfig)
    THINNING = ("thinning_config", ThinningConfig)
    SIMULATION = ("simulation_config", SimulationConfig)
    TRAINING = ("training_config", TrainingConfig)

    # Runner configurations
    RUNNER = ("runner_config", RunnerConfig)
    TRAINER = ("training_config", TrainingConfig)

    # HPO configurations
    HPO = ("hpo_config", HPOConfig)
    HPO_RUNNER = ("hpo_runner_config", HPORunnerConfig)

    # Logger configurations
    LOGGER = ("logger_config", LoggerConfig)

    @property
    def config_name(self) -> str:
        """Return the configuration name."""
        return self.value[0]

    def get_class(self) -> Any:
        """Return the Config class associated with this type."""
        return self.value[1]

    def get_class_name(self) -> str:
        """Return the name of the configuration class."""
        return self.get_class().__name__


class ConfigFactory:
    """Simple factory to create configuration instances."""

    def __init__(self):
        pass

    def create_config(
        self, config_type: ConfigType, config_data: dict, **kwargs
    ) -> Config:
        """
        Create a configuration instance.

        Args:
            config_type: The configuration type to create (enum)
            config_data: Configuration data (dict)
            **kwargs: Additional arguments for the constructor

        Returns:
            The configuration instance
        """
        config_class = config_type.get_class()
        config_name = config_type.get_class_name()

        logger.info(f"Creating configuration: {config_name}")

        try:
            # Create the instance directly
            instance = config_class(**config_data, **kwargs)

            # Additional validation
            if hasattr(instance, "validate"):
                instance.validate()

            return instance

        except Exception as e:
            logger.error(f"Error creating configuration {config_name}: {e}")
            raise ConfigValidationError(
                f"Failed to create {config_name} configuration: {str(e)}"
            ) from e

    def create_config_by_name(
        self, config_name: str, config_data: dict, **kwargs
    ) -> Config:
        """
        Create a configuration instance by name.

        Args:
            config_name: Configuration name (config_name or class_name)
            config_data: Configuration data
            **kwargs: Additional arguments

        Returns:
            The created configuration instance
        """
        # Trouver la configuration dans l'enum par nom de config ou nom de classe
        config_enum = None
        for config in ConfigType:
            if (
                config.config_name == config_name
                or config.get_class_name() == config_name
            ):
                config_enum = config
                break

        if config_enum is None:
            available_config_names = [c.config_name for c in ConfigType]
            available_class_names = [c.get_class_name() for c in ConfigType]
            raise ValueError(
                f"Configuration '{config_name}' not found.\n"
                f"Available config names: {available_config_names}\n"
                f"Available class names: {available_class_names}"
            )

        return self.create_config(config_enum, config_data, **kwargs)

    def list_available_configs(self) -> dict[str, str]:
        """List all available configurations with their config names and class names."""
        return {config.config_name: config.get_class_name() for config in ConfigType}

    def get_config_class(self, config_type: ConfigType) -> Type[Config]:
        """Get the class for a configuration type."""
        return config_type.get_class()

    def config_exists(self, config_name: str) -> bool:
        """Check if a configuration exists (by config name or class name)."""
        return any(
            config.config_name == config_name or config.get_class_name() == config_name
            for config in ConfigType
        )

    def create_from_yaml(
        self, config_type: ConfigType, yaml_path: str, **kwargs
    ) -> Config:
        """
        Create a configuration from a YAML file.

        Args:
            config_type: The configuration type to create
            yaml_path: Path to the YAML file
            **kwargs: Additional arguments

        Returns:
            The created configuration instance
        """
        from omegaconf import OmegaConf

        try:
            config_dict = dict(OmegaConf.load(yaml_path))
            return self.create_config(config_type, config_dict, **kwargs)
        except Exception as e:
            raise ConfigValidationError(
                f"Failed to load configuration from {yaml_path}: {str(e)}"
            ) from e


# Global factory instance
config_factory = ConfigFactory()
