"""
Factory simple pour les configurations TPP

Cette factory permet de créer facilement des instances de configurations
en utilisant un enum des configurations disponibles.

Utilisation:
    from easy_tpp.configs.config_factory import ConfigType, ConfigFactory

    factory = ConfigFactory()
    config = factory.create_config(ConfigType.MODEL, config_data)
"""

from enum import Enum
from typing import Type

from easy_tpp.configs.base_config import Config, ConfigValidationError
from easy_tpp.configs.data_config import DataConfig, DataLoadingSpecsConfig, TokenizerConfig
from easy_tpp.configs.hpo_config import HPOConfig, HPORunnerConfig
from easy_tpp.configs.logger_config import LoggerConfig
from easy_tpp.configs.model_config import (
    ModelConfig,
    ModelSpecsConfig,
    SimulationConfig,
    ThinningConfig,
    TrainingConfig,
)
from easy_tpp.configs.runner_config import RunnerConfig, TrainerConfig
from easy_tpp.utils import logger


class ConfigType(Enum):
    """Registre de tous les types de configurations disponibles sous forme (name, ConfigClass)."""
    
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
    TRAINER = ("trainer_config", TrainerConfig)
    
    # HPO configurations
    HPO = ("hpo_config", HPOConfig)
    HPO_RUNNER = ("hpo_runner_config", HPORunnerConfig)
    
    # Logger configurations
    LOGGER = ("logger_config", LoggerConfig)
    
    @property
    def config_name(self) -> str:
        """Obtenir le nom de la configuration."""
        return self.value[0]
    
    def get_class(self) -> Type[Config]:
        """Obtenir la classe de configuration associée à ce type."""
        return self.value[1]
    
    def get_class_name(self) -> str:
        """Obtenir le nom de la classe de configuration."""
        return self.get_class().__name__


class ConfigFactory:
    """Factory simple pour créer des instances de configurations."""

    def __init__(self):
        pass

    def create_config(
        self, config_type: ConfigType, config_data: dict, **kwargs
    ) -> Config:
        """
        Créer une instance de configuration.

        Args:
            config_type: Le type de configuration à créer (enum)
            config_data: Données de configuration (dict)
            **kwargs: Arguments additionnels pour le constructeur

        Returns:
            Instance de la configuration
        """
        config_class = config_type.get_class()
        config_name = config_type.get_class_name()

        logger.info(f"Création de la configuration: {config_name}")

        try:
            # Import validation utilities
            from easy_tpp.configs.config_utils import ConfigValidator
            
            # 1. Validate required fields if the class has this method
            if hasattr(config_class, '_get_required_fields_list'):
                required_fields = config_class._get_required_fields_list()
                ConfigValidator.validate_required_fields(
                    config_data, required_fields, config_class.__name__
                )
            
            # 2. Filter invalid fields
            filtered_data = ConfigValidator.filter_invalid_fields(config_data, config_class)
            
            # 3. Create the instance
            instance = config_class(**filtered_data, **kwargs)
            
            # 4. Additional validation
            if hasattr(instance, 'validate'):
                instance.validate()
            
            return instance

        except Exception as e:
            logger.error(f"Erreur lors de la création de la configuration {config_name}: {e}")
            raise ConfigValidationError(
                f"Failed to create {config_name} configuration: {str(e)}"
            ) from e

    def create_config_by_name(
        self, config_name: str, config_data: dict, **kwargs
    ) -> Config:
        """
        Créer une instance de configuration par nom.

        Args:
            config_name: Nom de la configuration (config_name ou class_name)
            config_data: Données de configuration
            **kwargs: Arguments additionnels

        Returns:
            Instance de la configuration
        """
        # Trouver la configuration dans l'enum par nom de config ou nom de classe
        config_enum = None
        for config in ConfigType:
            if config.config_name == config_name or config.get_class_name() == config_name:
                config_enum = config
                break

        if config_enum is None:
            available_config_names = [c.config_name for c in ConfigType]
            available_class_names = [c.get_class_name() for c in ConfigType]
            raise ValueError(
                f"Configuration '{config_name}' introuvable.\n"
                f"Noms de config disponibles: {available_config_names}\n"
                f"Noms de classes disponibles: {available_class_names}"
            )

        return self.create_config(config_enum, config_data, **kwargs)

    def list_available_configs(self) -> dict[str, str]:
        """Lister toutes les configurations disponibles avec leurs noms de config et noms de classes."""
        return {config.config_name: config.get_class_name() for config in ConfigType}

    def get_config_class(self, config_type: ConfigType) -> Type[Config]:
        """Obtenir la classe d'une configuration."""
        return config_type.get_class()

    def config_exists(self, config_name: str) -> bool:
        """Vérifier si une configuration existe (par nom de config ou nom de classe)."""
        return any(
            config.config_name == config_name or config.get_class_name() == config_name 
            for config in ConfigType
        )

    def create_from_yaml(
        self, config_type: ConfigType, yaml_path: str, **kwargs
    ) -> Config:
        """
        Créer une configuration à partir d'un fichier YAML.

        Args:
            config_type: Le type de configuration à créer
            yaml_path: Chemin vers le fichier YAML
            **kwargs: Arguments additionnels

        Returns:
            Instance de la configuration
        """
        from omegaconf import OmegaConf
        
        try:
            config_dict = OmegaConf.load(yaml_path)
            return self.create_config(config_type, config_dict, **kwargs)
        except Exception as e:
            raise ConfigValidationError(
                f"Failed to load configuration from {yaml_path}: {str(e)}"
            ) from e


# Instance globale de la factory
config_factory = ConfigFactory()