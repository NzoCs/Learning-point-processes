"""
Factory simple pour les modèles TPP

Cette factory permet de créer facilement des instances de modèles
en utilisant un enum des modèles disponibles.

Utilisation:
    from easy_tpp.models.model_factory import Models, ModelFactory

    factory = ModelFactory()
    model = factory.create_model(Models.NHP, model_config)
"""

from enum import Enum
from typing import Type

from easy_tpp.configs.model_config import ModelConfig
from easy_tpp.models.basemodel import BaseModel
from easy_tpp.models.model_registry import Models
from easy_tpp.utils import logger


class ModelFactory:
    """Factory simple pour créer des instances de modèles."""

    def __init__(self):
        pass

    def create_model(
        self, model: Models, model_config: ModelConfig, **kwargs
    ) -> BaseModel:
        """
        Créer une instance de modèle.

        Args:
            model: Le modèle à créer (enum)
            model_config: Configuration du modèle
            **kwargs: Arguments additionnels pour le constructeur

        Returns:
            Instance du modèle
        """
        model_class = model.get_class()
        model_name = model.get_class_name()

        logger.info(f"Création du modèle: {model_name}")

        try:
            instance = model_class(model_config, **kwargs)
            return instance

        except Exception as e:
            logger.error(f"Erreur lors de la création du modèle {model_name}: {e}")
            raise

    def create_model_by_name(
        self, model_name: str, model_config: ModelConfig, **kwargs
    ) -> BaseModel:
        """
        Créer une instance de modèle par nom.

        Args:
            model_name: Nom du modèle
            model_config: Configuration du modèle
            **kwargs: Arguments additionnels

        Returns:
            Instance du modèle
        """
        # Trouver le modèle dans l'enum par nom
        model_enum = None
        for model in Models:
            if model.get_class_name() == model_name:
                model_enum = model
                break

        if model_enum is None:
            available = [m.get_class_name() for m in Models]
            raise ValueError(
                f"Modèle '{model_name}' introuvable. Disponibles: {available}"
            )

        return self.create_model(model_enum, model_config, **kwargs)

    def list_available_models(self) -> list[str]:
        """Lister tous les modèles disponibles."""
        return [model.get_class_name() for model in Models]

    def get_model_class(self, model: Models) -> Type[BaseModel]:
        """Obtenir la classe d'un modèle."""
        return model.get_class()

    def model_exists(self, model_name: str) -> bool:
        """Vérifier si un modèle existe."""
        return any(model.get_class_name() == model_name for model in Models)
