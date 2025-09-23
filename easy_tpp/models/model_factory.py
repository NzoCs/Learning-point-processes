"""
Factory simple pour les modèles TPP

Cette factory permet de créer facilement des instances de modèles
en utilisant le ModelRegistry automatique.

Utilisation:
    from easy_tpp.models.model_factory import ModelFactory

    factory = ModelFactory()
    model = factory.create_model_by_name("NHP", model_config)
"""

from typing import Type

from easy_tpp.configs.model_config import ModelConfig
from easy_tpp.models.basemodel import Model
from easy_tpp.models.model_registry import ModelRegistry
from easy_tpp.utils import logger


class ModelFactory:
    """Factory simple pour créer des instances de modèles."""

    def __init__(self):
        pass

    def create_model_by_name(
        self, model_name: str, model_config: ModelConfig, **kwargs
    ) -> Model:
        """
        Créer une instance de modèle par nom.

        Args:
            model_name: Nom du modèle
            model_config: Configuration du modèle
            **kwargs: Arguments additionnels pour le constructeur

        Returns:
            Instance du modèle
        """
        logger.info(f"Création du modèle: {model_name}")

        # Récupérer la classe du modèle via le registry
        model_class = ModelRegistry.get_model(model_name)
        
        if model_class is None:
            available = ModelRegistry.list_models()
            if not available:
                logger.warning("Aucun modèle n'est enregistré. Assurez-vous d'importer les modèles avant d'utiliser la factory.")
            raise ValueError(
                f"Modèle '{model_name}' introuvable. Modèles disponibles: {available}"
            )

        try:
            instance = model_class(model_config, **kwargs)
            logger.debug(f"✅ Modèle '{model_name}' créé avec succès")
            return instance

        except Exception as e:
            logger.error(f"Erreur lors de la création du modèle {model_name}: {e}")
            raise

    def create_model(
        self, model_class: Type[Model], model_config: ModelConfig, **kwargs
    ) -> Model:
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
        logger.info(f"Création du modèle: {model_name}")

        try:
            instance = model_class(model_config, **kwargs)
            logger.debug(f"✅ Modèle '{model_name}' créé avec succès")
            return instance

        except Exception as e:
            logger.error(f"Erreur lors de la création du modèle {model_name}: {e}")
            raise

    def list_available_models(self) -> list[str]:
        """Lister tous les modèles disponibles."""
        return ModelRegistry.list_models()

    def get_model_class(self, model_name: str) -> Type[Model]:
        """Obtenir la classe d'un modèle par nom."""
        model_class = ModelRegistry.get_model(model_name)
        if model_class is None:
            available = ModelRegistry.list_models()
            raise ValueError(
                f"Modèle '{model_name}' introuvable. Modèles disponibles: {available}"
            )
        return model_class

    def model_exists(self, model_name: str) -> bool:
        """Vérifier si un modèle existe."""
        return ModelRegistry.model_exists(model_name)

    def get_registry(self) -> dict[str, Type[Model]]:
        """Obtenir le registry complet des modèles."""
        return ModelRegistry.get_registry()


# Instance globale de la factory
model_factory = ModelFactory()