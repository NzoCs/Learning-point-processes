"""
Automatic registry for TPP models

This module provides a registry that automatically registers new models
without needing manual additions.

Usage:
    from new_ltpp.models.model_registry import ModelRegistry

    # Get all registered models
    registry = ModelRegistry.get_registry()

    # Get a specific model
    model_class = ModelRegistry.get_model("NHP")
"""

from abc import ABCMeta
from typing import Dict, Optional, Type

from new_ltpp.utils import logger
from .model_protocol import ITPPModel


class ModelRegistry:
    """Automatic registry for TPP models."""

    _models: Dict[str, Type[ITPPModel]] = {}

    @classmethod
    def register_model(cls, name: str, model_class: Type[ITPPModel]) -> None:
        """
        Register a model in the registry.

        Args:
            name: Model name
            model_class: Model class
        """
        if name in cls._models:
            if cls._models[name] != model_class:
                logger.warning(
                    f"Warning: Model '{name}' already registered with a different class"
                )
            return

        cls._models[name] = model_class
        logger.debug(f"Model '{name}' auto-registered")

    @classmethod
    def get_registry(cls) -> Dict[str, Type[ITPPModel]]:
        """Get all registered models."""
        return cls._models.copy()

    @classmethod
    def get_model(cls, name: str) -> Optional[Type[ITPPModel]]:
        """
        Get a model by its name.

        Args:
            name: Model name

        Returns:
            Model class or None if not found
        """
        return cls._models.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def model_exists(cls, name: str) -> bool:
        """Check whether a model exists in the registry."""
        return name in cls._models


class RegistryMeta(ABCMeta):
    """Metaclass to automatically register new models.

    Inherits from ABCMeta to remain compatible with ABC.
    """

    def __new__(mcls, name: str, bases, namespace) -> Type[ITPPModel]:
        cls: Type[ITPPModel] = super().__new__(mcls, name, bases, namespace)  # type: ignore

        # Automatically register new model classes
        # Register everything except the Model base itself and plain Python base classes
        if name != "Model" and bases and bases != (object,):
            # Vérifier si on a une classe qui semble être un modèle TPP
            # (avoid registering utility/helper classes)
            is_model_class = any(
                hasattr(base, "__module__")
                and "new_ltpp.models" in getattr(base, "__module__", "")
                for base in bases
            )

            if is_model_class:
                ModelRegistry.register_model(name, cls)
                logger.debug(f"Model registered: {name}")

        return cls
