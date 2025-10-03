# Configuration Factory System

This module provides a centralized, type-safe configuration factory for EasyTPP that replaces repetitive `from_dict` methods with a unified approach.

## 🏭 Configuration Factory

The new `ConfigFactory` centralizes configuration creation logic and provides a type-safe, extensible way to create configuration instances.

### ✨ Key Features

- **Type Safety**: Uses enums and type hints for configuration types
- **Centralized Logic**: Single place for configuration creation logic
- **Validation**: Comprehensive validation with detailed error messages
- **Extensible**: Easy to register new configuration types
- **Backward Compatible**: Existing `from_dict` methods still work
- **YAML Support**: Direct loading from YAML files

## 🚀 Quick Start

### Basic Usage

```python
from new_ltpp.configs import ConfigType, config_factory

# Create a tokenizer configuration
tokenizer_data = {
    "num_event_types": 10,
    "padding_side": "left",
    "max_len": 512
}

config = config_factory.create_config(ConfigType.TOKENIZER, tokenizer_data)
```

### Available Configuration Types

```python
from new_ltpp.configs import config_factory

# Get all available configuration types
config_types = config_factory.get_available_config_types()
print(config_types)
# Output: ['tokenizer', 'data', 'model', 'training', ...]
```

### Using String Identifiers

```python
# You can use string identifiers instead of enums
model_config = config_factory.create_config("model", {
    "model_type": "NHP",
    "hidden_size": 128,
    "num_layers": 2
})
```

### Loading from YAML

```python
from new_ltpp.configs import config_factory

config = config_factory.create_from_yaml(ConfigType.MODEL, "path/to/config.yaml")
```

## 🔧 Advanced Usage

### Custom Validation

```python
# Create config with validation disabled
config = config_factory.create_config(ConfigType.TOKENIZER, data, validate=False)

# Enable validation (default)
config = config_factory.create_config(ConfigType.TOKENIZER, data, validate=True)
```

### Registering New Configuration Types

```python
from new_ltpp.configs import config_factory

# Register a custom configuration class
config_factory.register_config("my_custom_config", MyCustomConfigClass)
```

### Direct Factory Usage

```python
from new_ltpp.configs import config_factory

# Use the factory directly for more control
factory = config_factory
config = factory.create_config(ConfigType.MODEL, model_data)
```

## 📋 Configuration Types

| Type | Enum | Description |
|------|------|-------------|
| `TokenizerConfig` | `ConfigType.TOKENIZER` | Event tokenization settings |
| `DataConfig` | `ConfigType.DATA` | Data loading and preprocessing |
| `ModelConfig` | `ConfigType.MODEL` | Model architecture settings |
| `TrainingConfig` | `ConfigType.TRAINING` | Training parameters |
| `RunnerConfig` | `ConfigType.RUNNER` | Experiment runner settings |
| `HPOConfig` | `ConfigType.HPO` | Hyperparameter optimization |
| `LoggerConfig` | `ConfigType.LOGGER` | Logging configuration |

## 🔄 Migration from `from_dict`

### Old Pattern (still supported)
```python
from new_ltpp.configs.data_config import TokenizerConfig

config = TokenizerConfig.from_dict(config_dict)
```

### New Pattern (recommended)
```python
from new_ltpp.configs import ConfigType, config_factory

config = config_factory.create_config(ConfigType.TOKENIZER, config_dict)
```

## 🚫 Error Handling

The factory provides detailed error messages for common issues:

```python
from new_ltpp.configs import config_factory, ConfigType
from new_ltpp.configs.base_config import ConfigValidationError

try:
    config = config_factory.create_config(ConfigType.TOKENIZER, invalid_data)
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

## 📚 Examples

See `examples/config_factory_usage.py` for comprehensive usage examples including:

- Basic configuration creation
- Error handling
- YAML loading
- Migration patterns
- Advanced features

## 🏗️ Architecture

### Factory Pattern

```
ConfigFactory
├── _config_registry: Dict[ConfigType, Type[Config]]
├── create_config()
├── register_config()
└── create_from_yaml()
```

## 🎯 Best Practices

1. **Use Enums**: Prefer `ConfigType.TOKENIZER` over `"tokenizer"` strings
2. **Enable Validation**: Keep validation enabled unless you have specific reasons
3. **Handle Errors**: Always wrap configuration creation in try-catch blocks
4. **Document Custom Configs**: Add docstrings for custom configuration classes
5. **Use Factory Methods**: Use `config_factory.create_config()` for type-safe configuration creation