"""
Factory simple pour les modèles TPP

Cette factory permet de créer facilement des instances de modèles
en utilisant le ModelRegistry automatique.

Utilisation:
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
from .model_protocol import TPPModelProtocol


class ModelFactory:
    """Factory simple pour créer des instances de modèles."""

    def __init__(self):
        pass

    @staticmethod
    def create_model_by_name(
        model_name: str,
        model_config: ModelConfig,
        data_info: DataInfo,
        output_dir: Path | str,
        **kwargs,
    ) -> TPPModelProtocol:
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
                logger.warning(
                    "Aucun modèle n'est enregistré. Assurez-vous d'importer les modèles avant d'utiliser la factory."
                )
            raise ValueError(
                f"Modèle '{model_name}' introuvable. Modèles disponibles: {available}"
            )

        try:
            instance = model_class(
                model_config=model_config,
                data_info=data_info,
                output_dir=Path(output_dir),
                **model_config.specs.get_yaml_config(),
                **kwargs,
            )
            logger.debug(f"✅ Modèle '{model_name}' créé avec succès")
            return instance

        except Exception as e:
            logger.error(f"Erreur lors de la création du modèle {model_name}: {e}")
            raise

    @staticmethod
    def create_model(
        model_class: Type[TPPModelProtocol], model_config: ModelConfig, **kwargs
    ) -> TPPModelProtocol:
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
            model_specs = model_config.specs if hasattr(model_config, "specs") else {}
            model_specs_dict = (
                model_specs.get_yaml_config()
                if isinstance(model_specs, ModelSpecsConfig)
                else model_specs
            )
            instance = model_class(model_config, **model_specs_dict, **kwargs)
            logger.debug(f"✅ Modèle '{model_name}' créé avec succès")
            return instance

        except Exception as e:
            logger.error(f"Erreur lors de la création du modèle {model_name}: {e}")
            raise

    @staticmethod
    def list_available_models() -> list[str]:
        """Lister tous les modèles disponibles."""
        return ModelRegistry.list_models()

    @staticmethod
    def get_model_class(model_name: str) -> Type[TPPModelProtocol]:
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

    def get_registry(self) -> dict[str, Type[TPPModelProtocol]]:
        """Obtenir le registry complet des modèles."""
        return ModelRegistry.get_registry()


# Instance globale de la factory
model_factory = ModelFactory()
