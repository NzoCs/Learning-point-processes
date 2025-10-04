"""
Interface de configuration pour éviter les imports circulaires

Ce module définit les interfaces et protocoles de base pour les configurations,
permettant d'éviter les imports circulaires entre les différents modules de configuration.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Protocol


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigSerializationError(Exception):
    """Raised when configuration serialization/deserialization fails."""

    pass


class ConfigInterface(Protocol):
    """
    Protocol définissant l'interface de base pour toutes les configurations.

    Ce protocol permet aux autres modules d'utiliser les configurations
    sans importer les classes concrètes, évitant ainsi les imports circulaires.
    """

    def get_required_fields(self) -> List[str]:
        """Return list of required field names."""
        ...

    def get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration as YAML-compatible dictionary."""
        ...

    def validate(self) -> None:
        """Validate the configuration."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        ...
