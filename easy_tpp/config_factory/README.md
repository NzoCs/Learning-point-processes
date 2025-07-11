# Configuration Factory for EasyTPP

This directory contains the configuration management system for the EasyTPP framework. It provides type-safe, validated configuration classes for all components.

## Overview

The configuration factory provides:

- **Type Safety**: Dataclass-based configurations with type hints
- **Validation**: Comprehensive validation with clear error messages
- **Serialization**: Support for YAML, JSON, and dictionary formats
- **Integration**: Seamless integration with all EasyTPP components

## Directory Structure

```
config_factory/
├── base.py              # Base configuration classes and interfaces
├── data_config.py       # Data loading and preprocessing configurations
├── model_config.py      # Model architecture and training configurations
├── runner_config.py     # Training pipeline and experiment configurations
├── hpo_config.py        # Hyperparameter optimization configurations
├── logger_config.py     # Logging and experiment tracking configurations
└── __init__.py          # Module exports and factories
```

## Quick Start

### Basic Configuration

```python
from easy_tpp.config_factory import DataConfig, ModelConfig, RunnerConfig

# Create data configuration
data_config = DataConfig(
    data_dir='./data/synthetic',
    data_format='json',
    data_specs={
        'num_event_types': 3,
        'max_seq_len': 100
    },
    data_loading_specs={
        'batch_size': 32,
        'shuffle': True
    }
)

# Create model configuration
model_config = ModelConfig(
    model_type='NHP',
    model_specs={
        'hidden_size': 64,
        'num_layers': 2
    }
)
```

### Configuration from Files

```python
from easy_tpp.config_factory import config_factory

# Load from YAML file
config = config_factory.from_yaml('config.yaml')

# Load from dictionary  
config = config_factory.from_dict(config_dict)
```

## Configuration Types

### DataConfig
Configuration for data loading and preprocessing:

```python
data_config = DataConfig(
    data_dir='./data/',
    data_format='json',
    data_specs=TokenizerConfig(num_event_types=5),
    data_loading_specs=DataLoadingSpecsConfig(batch_size=64)
)
```

### ModelConfig  
Configuration for model architecture and training:

```python
model_config = ModelConfig(
    model_type=ModelType.NHP,
    model_specs={'hidden_size': 128},
    training_specs={'learning_rate': 1e-3}
)
```

### RunnerConfig
Configuration for training pipeline:

```python
runner_config = RunnerConfig(
    dataset_id='experiment_1',
    model_id='nhp_baseline',
    max_epochs=100,
    logger_config=LoggerConfig(logger_type=LoggerType.WandB)
)
```

### HPOConfig
Configuration for hyperparameter optimization:

```python
hpo_config = HPOConfig(
    framework_id='optuna',
    num_trials=100,
    storage_uri='sqlite:///study.db'
)
```

### LoggerConfig
Configuration for experiment tracking:

```python
logger_config = LoggerConfig(
    logger_type=LoggerType.WandB,
    config={
        'project': 'tpp_experiments',
        'name': 'experiment_1'
    }
)
```

## Validation System

All configurations include built-in validation:

```python
try:
    config = DataConfig(
        data_dir='',  # Invalid empty path
        batch_size=-1  # Invalid negative value
    )
    config.validate()
except ConfigValidationError as e:
    print(f"Validation failed: {e}")
```

## YAML Configuration

Example configuration file:

```yaml
# config.yaml
data:
  data_dir: "./data/synthetic"
  data_format: "json"
  data_specs:
    num_event_types: 3
  data_loading_specs:
    batch_size: 64

model:
  model_type: "NHP"
  model_specs:
    hidden_size: 128
  
runner:
  dataset_id: "experiment_1"
  model_id: "nhp_baseline"
  max_epochs: 100
```

Load with:

```python
config = config_factory.from_yaml('config.yaml')
```

## Available Components

- **BaseConfig**: Foundation class for all configurations
- **ConfigFactory**: Central factory for creating configurations
- **ConfigValidator**: Extensible validation system
- **TokenizerConfig**: Event tokenization settings
- **DataLoadingSpecsConfig**: Data loading specifications
- **ThinningConfig**: Thinning process configuration
- **LoggerConfig**: Experiment tracking configuration

## Usage Examples

### Complete Configuration Setup

```python
# Load base configuration
config = config_factory.from_yaml('base_config.yaml')

# Access components
data_module = TPPDataModule(config.data_config)
model = create_model(config.model_config)
trainer = create_trainer(config.runner_config)
```

### Configuration Export

```python
# Export to YAML
yaml_config = config.to_yaml()

# Export to dictionary
config_dict = config.to_dict()
```