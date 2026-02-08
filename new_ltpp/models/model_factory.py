"""
Simple factory for TPP models

This factory makes it easy to create model instances
using the automatic ModelRegistry.

Usage:
    from new_ltpp.models.model_factory import ModelFactory

    factory = ModelFactory()
    model = factory.create_model_by_name("NHP", model_config)
"""

from pathlib import Path
from typing import Type

from new_ltpp.configs import ModelConfig, ModelSpecsConfig
from new_ltpp.shared_types import DataInfo
from new_ltpp.utils import logger

from .model_registry import ModelRegistry
from .model_protocol import ITPPModel


class ModelFactory:
    """Simple factory to create model instances."""

    def __init__(self):
        pass

    @staticmethod
    def create_model_by_name(
        model_name: str,
        model_config: ModelConfig,
        data_info: DataInfo,
        output_dir: Path | str,
        **kwargs,
    ) -> ITPPModel:
        """
        Créer une instance de modèle par nom.

        Args:
            model_name: Nom du modèle
            model_config: Configuration du modèle
            **kwargs: Arguments additionnels pour le constructeur

        Returns:
            Instance du modèle
        """

        logger.info(f"Creating model: {model_name}")

        # Récupérer la classe du modèle via le registry
        model_class = ModelRegistry.get_model(model_name)

        if model_class is None:
            available = ModelRegistry.list_models()
            if not available:
                logger.warning(
                    "No models are registered. Ensure models are imported before using the factory."
                )
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available}"
            )

        try:
            instance = model_class(
                model_config=model_config,
                data_info=data_info,
                output_dir=Path(output_dir),
                **model_config.specs.get_yaml_config(),
                **kwargs,
            )
            logger.debug(f"Model '{model_name}' created successfully")
            return instance

        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
            raise

    @staticmethod
    def create_model(
        model_class: Type[ITPPModel], model_config: ModelConfig, **kwargs
    ) -> ITPPModel:
        """
        Créer une instance de modèle directement avec la classe.

        Args:
            model_class: Classe du modèle
            model_config: Configuration du modèle
            **kwargs: Arguments additionnels pour le constructeur

        Returns:
            Instance du modèle
        """
        model_name = model_class.__name__
        logger.info(f"Creating model: {model_name}")

        try:
            model_specs = model_config.specs if hasattr(model_config, "specs") else {}
            model_specs_dict = (
                model_specs.get_yaml_config()
                if isinstance(model_specs, ModelSpecsConfig)
                else model_specs
            )
            instance = model_class(model_config, **model_specs_dict, **kwargs)
            logger.debug(f"Model '{model_name}' created successfully")
            return instance

        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
            raise

    @staticmethod
    def list_available_models() -> list[str]:
        """List all available models."""
        return ModelRegistry.list_models()

    @staticmethod
    def get_model_class(model_name: str) -> Type[ITPPModel]:
        """Get the class of a model by name."""
        model_class = ModelRegistry.get_model(model_name)
        if model_class is None:
            available = ModelRegistry.list_models()
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available}"
            )
        return model_class

    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists."""
        return ModelRegistry.model_exists(model_name)

    def get_registry(self) -> dict[str, Type[ITPPModel]]:
        """Get the full models registry."""
        return ModelRegistry.get_registry()


# Global factory instance
model_factory = ModelFactory()
