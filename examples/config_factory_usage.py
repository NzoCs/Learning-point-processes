"""
Example usage of the new Configuration Factory System.

This file demonstrates how to use the new ConfigFactory to create
configuration instances from dictionaries, replacing the old from_dict methods.
"""

from easy_tpp.configs import (
    ConfigType,
    config_factory,
)


def example_basic_usage():
    """Example of basic factory usage."""
    
    print("=== Configuration Factory Examples ===\n")
    
    # 1. Create a TokenizerConfig using enum
    tokenizer_data = {
        "num_event_types": 10,
        "padding_side": "left",
        "truncation_side": "right",
        "max_len": 512
    }
    
    tokenizer_config = config_factory.create_config(ConfigType.TOKENIZER, tokenizer_data)
    print(f"1. Tokenizer config created: {tokenizer_config}")
    print(f"   Type: {type(tokenizer_config).__name__}")
    print(f"   Pad token ID: {tokenizer_config.pad_token_id}\n")
    
    # 2. Create a ModelConfig using string
    model_data = {
        "model_type": "NHP",
        "hidden_size": 128,
        "num_layers": 2,
        "num_event_types": 10
    }
    
    model_config = config_factory.create_config("model", model_data)
    print(f"2. Model config created: {type(model_config).__name__}")
    print(f"   Model type: {getattr(model_config, 'model_type', 'N/A')}")
    print(f"   Hidden size: {getattr(model_config, 'hidden_size', 'N/A')}\n")
    
    # 3. Create a TrainingConfig using enum
    training_data = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 100,
        "optimizer": "Adam"
    }
    
    training_config = config_factory.create_config(ConfigType.TRAINING, training_data)
    print(f"3. Training config created: {type(training_config).__name__}")
    print(f"   Learning rate: {getattr(training_config, 'learning_rate', 'N/A')}")
    print(f"   Optimizer: {getattr(training_config, 'optimizer', 'N/A')}\n")


def example_error_handling():
    """Example of error handling with the factory."""
    
    print("=== Error Handling Examples ===\n")
    
    # 1. Invalid config type
    try:
        invalid_config = config_factory.create_config("invalid_type", {})
    except Exception as e:
        print(f"1. Expected error for invalid type: {type(e).__name__}: {e}\n")
    
    # 2. Missing required fields (if validation is implemented)
    try:
        incomplete_data = {}  # Missing required fields
        config = config_factory.create_config(ConfigType.TOKENIZER, incomplete_data)
    except Exception as e:
        print(f"2. Validation error: {type(e).__name__}: {e}\n")


def example_advanced_usage():
    """Example of advanced factory features."""
    
    print("=== Advanced Usage Examples ===\n")
    
    # 1. Get available config types
    available_types = config_factory.get_available_config_types()
    print(f"1. Available config types: {available_types}\n")
    
    # 2. Create config without validation
    risky_data = {
        "num_event_types": -1,  # Invalid value
        "unknown_field": "value"  # Unknown field
    }
    
    try:
        config = config_factory.create_config(ConfigType.TOKENIZER, risky_data, validate=False)
        print(f"2. Config created without validation: {type(config).__name__}")
        print(f"   Note: This may contain invalid values!\n")
    except Exception as e:
        print(f"2. Even without validation, creation failed: {e}\n")


def example_migration_from_old_pattern():
    """Example showing migration from old from_dict pattern."""
    
    print("=== Migration Example ===\n")
    
    config_data = {
        "num_event_types": 5,
        "padding_side": "left",
        "max_len": 256
    }
    
    # OLD WAY (still works for backward compatibility)
    print("Old way:")
    try:
        from easy_tpp.configs.data_config import TokenizerConfig
        old_config = TokenizerConfig.from_dict(config_data)
        print(f"  TokenizerConfig.from_dict() -> {type(old_config).__name__}")
    except Exception as e:
        print(f"  Old way failed: {e}")
    
    # NEW WAY (recommended)
    print("New way:")
    new_config = config_factory.create_config(ConfigType.TOKENIZER, config_data)
    print(f"  create_config(ConfigType.TOKENIZER, data) -> {type(new_config).__name__}")
    print(f"  Same result: {old_config.__dict__ == new_config.__dict__}")


def example_yaml_loading():
    """Example of loading configuration from YAML file."""
    
    print("=== YAML Loading Example ===\n")
    
    # This would work if you have a YAML file
    yaml_example = """
# Example YAML that could be loaded:
# tokenizer_config.yaml:
# num_event_types: 10
# padding_side: "left"
# truncation_side: "left"
# max_len: 512

# Usage:
# config = config_factory.create_from_yaml(ConfigType.TOKENIZER, "tokenizer_config.yaml")
"""
    print(yaml_example)


if __name__ == "__main__":
    """Run all examples."""
    
    try:
        example_basic_usage()
        example_error_handling()
        example_advanced_usage()
        example_migration_from_old_pattern()
        example_yaml_loading()
        
        print("🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Example failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()