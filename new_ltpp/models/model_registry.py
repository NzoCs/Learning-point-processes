"""
Registry automatique pour les modèles TPP

Ce module fournit un registry qui enregistre automatiquement tous les nouveaux modèles
sans avoir besoin de les ajouter manuellement.

Utilisation:
    from new_ltpp.models.model_registry import ModelRegistry

    # Obtenir tous les modèles enregistrés
    registry = ModelRegistry.get_registry()

    # Obtenir un modèle spécifique
    model_class = ModelRegistry.get_model("NHP")
"""

from abc import ABCMeta
from typing import Dict, Optional, Type

from new_ltpp.utils import logger


class ModelRegistry:
    """Registry automatique pour les modèles TPP."""

    _models: Dict[str, Type] = {}

    @classmethod
    def register_model(cls, name: str, model_class: Type) -> None:
        """
        Enregistrer un modèle dans le registry.

        Args:
            name: Nom du modèle
            model_class: Classe du modèle
        """
        if name in cls._models:
            if cls._models[name] != model_class:
                logger.warning(
                    f"⚠️ Modèle '{name}' déjà enregistré avec une classe différente"
                )
            return

        cls._models[name] = model_class
        logger.debug(f"✅ Modèle '{name}' enregistré automatiquement")

    @classmethod
    def get_registry(cls) -> Dict[str, Type]:
        """Obtenir tous les modèles enregistrés."""
        return cls._models.copy()

    @classmethod
    def get_model(cls, name: str) -> Optional[Type]:
        """
        Obtenir un modèle par son nom.

        Args:
            name: Nom du modèle

        Returns:
            Classe du modèle ou None si non trouvé
        """
        return cls._models.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        """Lister tous les noms de modèles enregistrés."""
        return list(cls._models.keys())

    @classmethod
    def model_exists(cls, name: str) -> bool:
        """Vérifier si un modèle existe dans le registry."""
        return name in cls._models


class RegistryMeta(ABCMeta):
    """Metaclasse pour enregistrer automatiquement les nouveaux modèles.

    Hérite de ABCMeta pour être compatible avec ABC.
    """

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        # Enregistrer automatiquement les nouvelles classes de modèles
        # On enregistre tout sauf Model lui-même et les classes de base Python
        if name != "Model" and bases and bases != (object,):
            # Vérifier si on a une classe qui semble être un modèle TPP
            # (éviter d'enregistrer des classes utilitaires)
            is_model_class = any(
                hasattr(base, "__module__")
                and "new_ltpp.models" in getattr(base, "__module__", "")
                for base in bases
            )

            if is_model_class:
                ModelRegistry.register_model(name, cls)
                logger.debug(f"Modèle enregistré: {name}")

        return cls
