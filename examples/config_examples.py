"""
Example usage of configuration transformers and factory.

This module demonstrates how to use the ConfigTransformer utilities
to transform raw YAML configurations before creating configuration instances.
"""

from typing import Dict, Any
from easy_tpp.configs.config_utils import ConfigTransformer
from easy_tpp.configs.runner_config import RunnerConfig
from easy_tpp.configs.model_config import ModelConfig, TrainingConfig
from easy_tpp.configs.data_config import DataConfig
from easy_tpp.configs.logger_config import LoggerConfig


def create_runner_config_from_yaml(yaml_config: Dict[str, Any]) -> RunnerConfig:
    """
    Example function showing how to create a RunnerConfig from raw YAML configuration.
    
    This function demonstrates the proper separation of concerns:
    1. Transform the raw YAML configuration
    2. Use the clean from_dict methods to create instances
    
    Args:
        yaml_config: Raw configuration dictionary from YAML
        
    Returns:
        RunnerConfig instance
    """
    # 1. Transform the raw configuration to handle legacy formats and type conversions
    transformed_config = ConfigTransformer.transform_runner_config(yaml_config)
    
    # 2. Create the configuration instance using the clean from_dict method
    return RunnerConfig.from_dict(transformed_config)


def create_model_config_with_legacy_support(yaml_config: Dict[str, Any]) -> ModelConfig:
    """
    Example showing how to handle legacy model configuration formats.
    
    Args:
        yaml_config: Raw configuration dictionary that might contain legacy fields
        
    Returns:
        ModelConfig instance
    """
    # Transform to handle legacy field names and format conversions
    transformed_config = ConfigTransformer.transform_model_config(yaml_config)
    
    # Create using the clean from_dict method
    return ModelConfig.from_dict(transformed_config)


def create_training_config_with_aliases(yaml_config: Dict[str, Any]) -> TrainingConfig:
    """
    Example showing how to handle training configuration with field aliases.
    
    Args:
        yaml_config: Raw configuration with potential alias fields like 'dropout_rate'
        
    Returns:
        TrainingConfig instance
    """
    # Transform to handle aliases like dropout_rate -> dropout
    transformed_config = ConfigTransformer.transform_training_config(yaml_config)
    
    # Create using the clean from_dict method
    return TrainingConfig.from_dict(transformed_config)


def create_data_config_with_directory_handling(yaml_config: Dict[str, Any]) -> DataConfig:
    """
    Example showing how to handle data configuration with flexible directory structures.
    
    Args:
        yaml_config: Raw configuration that might use 'source_dir' or split directories
        
    Returns:
        DataConfig instance
    """
    # Transform to handle directory structure conversion
    transformed_config = ConfigTransformer.transform_data_config(yaml_config)
    
    # Create using the clean from_dict method
    return DataConfig.from_dict(transformed_config)


def create_logger_config_with_flexible_format(yaml_config: Dict[str, Any]) -> LoggerConfig:
    """
    Example showing how to handle logger configuration with flexible key names.
    
    Args:
        yaml_config: Raw configuration that might use 'type' instead of 'logger_type'
        
    Returns:
        LoggerConfig instance
    """
    # Transform to handle flexible field names
    transformed_config = ConfigTransformer.transform_logger_config(yaml_config)
    
    # Create using the clean from_dict method
    return LoggerConfig.from_dict(transformed_config)


# Example of how to extend transformers for new formats
class CustomConfigTransformer:
    """
    Example of how to extend the transformation logic for custom formats.
    """
    
    @staticmethod
    def transform_custom_runner_format(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example transformation for a custom runner configuration format.
        
        This could handle a completely different YAML structure that needs
        to be converted to the standard format.
        """
        # Example: Convert from nested format to flat format
        if "training" in yaml_config and "model" in yaml_config["training"]:
            # Convert nested structure
            config = dict(yaml_config)
            config["trainer_config"] = yaml_config["training"].get("trainer", {})
            config["model_config"] = yaml_config["training"]["model"]
            config["data_config"] = yaml_config.get("data", {})
            
            # Remove old keys
            config.pop("training", None)
            
            # Apply standard transformations
            return ConfigTransformer.transform_runner_config(config)
        
        # Fall back to standard transformation
        return ConfigTransformer.transform_runner_config(yaml_config)


def example_usage():
    """
    Example demonstrating the usage patterns.
    """
    # Example 1: Standard usage with transformation
    yaml_config = {
        "trainer_config": {
            "dataset_id": "synthetic",
            "model_id": "NHP",
            "batch_size": 32,
            "dropout_rate": 0.1,  # Legacy field name
            "logger_config": {
                "type": "wandb",  # Alternative field name
                "project": "my_project"
            }
        },
        "model_config": {
            "model_id": "NHP",
            "num_event_types": 5,
            "base_config": {
                "backend": "torch"  # String instead of enum
            }
        },
        "data_config": {
            "source_dir": "/path/to/data"  # Legacy single directory
        }
    }
    
    # Transform and create
    runner_config = create_runner_config_from_yaml(yaml_config)
    print(f"Created runner config: {runner_config}")
    
    # Example 2: Direct transformation for specific configs
    model_yaml = {
        "model_id": "RMTPP",
        "num_event_types": 10,
        "specs": {
            "hidden_size": 128,
            "rnn_type": "LSTM"
        }
    }
    
    model_config = create_model_config_with_legacy_support(model_yaml)
    print(f"Created model config: {model_config}")


if __name__ == "__main__":
    example_usage()
