"""
Configuration utilities for handling YAML to dict transformations and field mappings.

This module contains utilities for transforming YAML configurations into dictionaries
that are ready to be validated and converted to configuration instances. It handles
legacy field mappings, type conversions, and other preprocessing tasks.
"""

from typing import Callable, List

from new_ltpp.configs.base_config import Config, ConfigValidationError


class ConfigValidator:
    """
    Configuration validator with extensible validation rules.

    Provides a framework for validating configuration objects with
    clear error reporting and customizable validation rules.
    """

    def __init__(self):
        self._validation_rules: List[Callable[[Config], None]] = []

    def add_rule(self, rule_func: Callable[[Config], None]) -> None:
        """Add a validation rule function."""
        self._validation_rules.append(rule_func)

    def validate(self, config: Config) -> List[str]:
        """
        Validate configuration and return list of error messages.

        Args:
            config: Configuration object to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for rule in self._validation_rules:
            try:
                rule(config)
            except ConfigValidationError as e:
                error_msg = f"{e.field_name}: {str(e)}" if e.field_name else str(e)
                errors.append(error_msg)
            except Exception as e:
                errors.append(f"Validation error: {str(e)}")

        return errors

    def validate_required_fields(
        self, config: Config, required_fields: List[str]
    ) -> None:
        """Validate that required fields are present and not None."""
        for field_name in required_fields:
            if not hasattr(config, field_name):
                raise ConfigValidationError(
                    f"Required field '{field_name}' is missing", field_name
                )

            value = getattr(config, field_name)
            if value is None:
                raise ConfigValidationError(
                    f"Required field '{field_name}' cannot be None",
                    field_name,
                )
