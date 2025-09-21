"""
Configuration utilities for handling YAML to dict transformations and field mappings.

This module contains utilities for transforming YAML configurations into dictionaries
that are ready to be validated and converted to configuration instances. It handles
legacy field mappings, type conversions, and other preprocessing tasks.
"""

import logging
from dataclasses import fields
from typing import Any, Dict, List, Optional, Type, Union

from easy_tpp.utils.const import Backend

logger = logging.getLogger(__name__)


class ConfigTransformer:
    """
    Handles transformation of raw configuration dictionaries into validated,
    standardized dictionaries ready for configuration object creation.
    """

    @staticmethod
    def transform_trainer_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw trainer configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        config = dict(config_dict)

        # Handle dropout_rate alias
        if "dropout_rate" in config and "dropout" not in config:
            config["dropout"] = config["dropout_rate"]

        # Handle logger config preprocessing only if logging is activated
        if config.get("activate_logging", False):
            logger_cfg = config.get("logger_config", {})
            if (
                not hasattr(logger_cfg, "__class__")
                or logger_cfg.__class__.__name__ != "LoggerConfig"
            ):
                # Calculate save_dir for logger config
                ckpt = config.get("checkpoint_dir", "checkpoints")
                model_id = config.get("model_id", "unknown")
                dataset_id = config.get("dataset_id", "unknown")
                save_dir = (
                    config.get("save_dir") or f"./{ckpt}/{model_id}/{dataset_id}/"
                )

                config["logger_config"] = ConfigTransformer._prepare_logger_config(
                    logger_cfg, save_dir=save_dir
                )
        elif "logger_config" in config:
            # If logging is not activated, remove logger_config
            config.pop("logger_config", None)

        return config

    @staticmethod
    def transform_model_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw model configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        config = dict(config_dict)

        # Transform sub-configurations
        sub_configs = {
            "base_config": "training_config",
            "specs": "model_specs_config",
            "thinning": "thinning_config",
            "simulation_config": "simulation_config",
        }

        for field_name, config_type in sub_configs.items():
            if field_name in config and isinstance(config[field_name], dict):
                if config_type == "training_config":
                    config[field_name] = ConfigTransformer.transform_training_config(
                        config[field_name]
                    )
                elif config_type == "model_specs_config":
                    config[field_name] = ConfigTransformer.transform_model_specs_config(
                        config[field_name]
                    )
                # thinning_config and simulation_config don't need special transformation

        return config

    @staticmethod
    def transform_training_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw training configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        config = dict(config_dict)

        # Handle dropout_rate alias
        if "dropout_rate" in config and "dropout" not in config:
            config["dropout"] = config["dropout_rate"]

        # Handle backend conversion
        backend = config.get("backend", Backend.Torch)
        if isinstance(backend, str):
            backend_lower = backend.lower()
            if backend_lower in ["torch", "pytorch"]:
                config["backend"] = Backend.Torch
            elif backend_lower in ["tf", "tensorflow"]:
                config["backend"] = Backend.TF
            else:
                logger.warning(f"Unknown backend: {backend}, defaulting to Torch")
                config["backend"] = Backend.Torch
        else:
            config["backend"] = backend

        return config

    @staticmethod
    def transform_model_specs_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw model specs configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        # Model specs config currently doesn't need special transformation
        return dict(config_dict)

    @staticmethod
    def transform_data_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw data configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        config = dict(config_dict)

        # Handle backward compatibility for directory structure
        train_dir = config.get("train_dir")
        valid_dir = config.get("valid_dir")
        test_dir = config.get("test_dir")
        source_dir = config.get("source_dir")

        # Validate directory parameters
        has_split_dirs = any([train_dir, valid_dir, test_dir])
        if has_split_dirs and source_dir:
            raise ValueError(
                "Cannot specify both split directories (train_dir, valid_dir, test_dir) and source_dir"
            )

        # Set directories based on provided parameters
        if train_dir and valid_dir and test_dir:
            # All three directories provided - use them directly
            pass
        elif source_dir:
            # Only source_dir provided - use it for all three
            logger.warning(
                f"Only source_dir provided ({source_dir}). Using it for train_dir, valid_dir, and test_dir"
            )
            config["train_dir"] = source_dir
            config["valid_dir"] = source_dir
            config["test_dir"] = source_dir
        elif train_dir or valid_dir or test_dir:
            # Partial directories provided - require all three
            raise ValueError(
                "When providing split directories, all three (train_dir, valid_dir, test_dir) must be specified"
            )
        else:
            raise ValueError(
                "Either provide split directories (train_dir, valid_dir, test_dir) or a source_dir"
            )

        # Transform sub-configurations
        dls = config.get("data_loading_specs", {})
        if isinstance(dls, dict):
            config["data_loading_specs"] = (
                ConfigTransformer.transform_data_loading_specs_config(dls)
            )

        ds = config.get("data_specs", {})
        if isinstance(ds, dict):
            config["data_specs"] = ConfigTransformer.transform_tokenizer_config(ds)

        return config

    @staticmethod
    def transform_data_loading_specs_config(
        config_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transform raw data loading specs configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        # Data loading specs config currently doesn't need special transformation
        return dict(config_dict)

    @staticmethod
    def transform_tokenizer_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw tokenizer configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        # Tokenizer config currently doesn't need special transformation
        return dict(config_dict)

    @staticmethod
    def transform_logger_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw logger configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        config = dict(config_dict)

        # Handle legacy formats where 'type' might be used instead of 'logger_type'
        logger_type = config.get("logger_type") or config.get("type")
        if logger_type:
            config["logger_type"] = logger_type

        # Handle config field - merge additional parameters
        config_keys = {"logger_type", "save_dir", "config"}
        additional_config = {k: v for k, v in config.items() if k not in config_keys}
        if additional_config:
            config["config"] = {**config.get("config", {}), **additional_config}

        return config

    @staticmethod
    def transform_runner_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw runner configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Transformed and validated dictionary
        """
        config = dict(config_dict)

        # Transform sub-configurations
        if "trainer_config" in config and isinstance(config["trainer_config"], dict):
            config["trainer_config"] = ConfigTransformer.transform_trainer_config(
                config["trainer_config"]
            )

        if "model_config" in config and isinstance(config["model_config"], dict):
            config["model_config"] = ConfigTransformer.transform_model_config(
                config["model_config"]
            )

        if "data_config" in config and isinstance(config["data_config"], dict):
            config["data_config"] = ConfigTransformer.transform_data_config(
                config["data_config"]
            )

        return config

    @staticmethod
    def _prepare_logger_config(
        logger_cfg: Dict[str, Any], save_dir: str
    ) -> Dict[str, Any]:
        """
        Prepare logger configuration dictionary with save_dir.

        Args:
            logger_cfg: Raw logger configuration
            save_dir: Directory for saving logs

        Returns:
            Prepared logger configuration
        """
        prepared = dict(logger_cfg)
        prepared["save_dir"] = save_dir
        return ConfigTransformer.transform_logger_config(prepared)


class ConfigValidator:
    """
    Handles validation of configuration dictionaries before object creation.
    """

    @staticmethod
    def validate_required_fields(
        config_dict: Dict[str, Any], required_fields: List[str], config_name: str
    ) -> None:
        """
        Validate that required fields are present in the configuration dictionary.

        Args:
            config_dict: Configuration dictionary to validate
            required_fields: List of required field names
            config_name: Name of the configuration for error messages

        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = []
        for field in required_fields:
            if field not in config_dict or config_dict[field] is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Missing required fields in {config_name}: {missing_fields}"
            )

    @staticmethod
    def filter_invalid_fields(
        config_dict: Dict[str, Any], config_class: Type
    ) -> Dict[str, Any]:
        """
        Filter out invalid fields from configuration dictionary.

        Args:
            config_dict: Configuration dictionary to filter
            config_class: Configuration class to check valid fields against

        Returns:
            Filtered configuration dictionary
        """
        if not hasattr(config_class, "__dataclass_fields__"):
            return config_dict

        valid_keys = set(config_class.__dataclass_fields__.keys())
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        dropped = set(config_dict.keys()) - valid_keys

        if dropped:
            logger.warning(
                f"Filtered out invalid {config_class.__name__} keys: {dropped}"
            )

        return filtered
