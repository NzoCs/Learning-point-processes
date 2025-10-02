# Data Preprocessing for Temporal Point Processes

This directory contains the data preprocessing pipeline for temporal point process datasets in the EasyTPP framework.

## Overview

The preprocessing module provides comprehensive tools for loading, tokenizing, collating, and visualizing temporal point process data. It supports multiple data formats and integrates seamlessly with PyTorch Lightning for efficient data handling.

## Available Modules

- `dataset.py`: Core dataset class for temporal point process data
- `data_loader.py`: PyTorch Lightning DataModule for efficient data loading
- `data_collator.py`: Data collation utilities for batching sequences
- `event_tokenizer.py`: Event tokenization and encoding functionality
- `visualizer.py`: Data visualization and analysis tools
- `__init__.py`: Module exports (`TPPDataModule`, `TPPDataset`, `EventTokenizer`)

## Table of Contents

- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Data Loading](#data-loading)
- [Event Tokenization](#event-tokenization)
- [Data Visualization](#data-visualization)
- [Configuration](#configuration)
- [Examples](#examples)
- [Data Formats](#data-formats)

## Quick Start

### Basic Data Loading

```python
from easy_tpp.data.preprocess import TPPDataModule
from easy_tpp.config_factory import DataConfig

# Create data configuration
data_config = DataConfig(
    data_dir='./data/synthetic',
    data_format='json',
    tokenizer_specs={
        'num_event_types': 3,
        'max_seq_len': 50
    },
    data_loading_specs={
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4
    }
)

# Initialize data module
data_module = TPPDataModule(data_config)

# Setup data loaders
data_module.setup()

# Get data loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

### Simple Visualization

```python
from easy_tpp.data.preprocess import Visualizer

# Create visualizer
visualizer = Visualizer(
    data_module=data_module,
    split='test',
    save_dir='./plots/',
    dataset_size=1000
)

# Generate basic statistics
visualizer.plot_event_type_distribution()
visualizer.plot_inter_event_time_distribution()
visualizer.plot_sequence_length_distribution()
```

## Core Components

### 1. TPPDataset

The core dataset class that handles temporal point process sequences.

```python
from easy_tpp.data.preprocess import TPPDataset

# Data should contain time_seqs, time_delta_seqs, and type_seqs
dataset = TPPDataset(data={
    'time_seqs': [...],      # Event timestamps
    'time_delta_seqs': [...], # Inter-event times
    'type_seqs': [...]       # Event types
})

# Access data
sample = dataset[0]
print(f"Sequence length: {len(sample['time_seqs'])}")
print(f"Event types: {sample['type_seqs']}")
```

### 2. TPPDataModule (PyTorch Lightning)

Comprehensive data module supporting multiple formats and efficient loading.

```python
from easy_tpp.data.preprocess import TPPDataModule

class TPPDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for temporal point processes.
    
    Features:
    - Support for JSON and PKL data formats
    - Automatic data validation
    - Built-in tokenization and collation
    - Configurable padding and truncation
    """
```

**Key Methods:**

- `setup()`: Prepare datasets for training/validation/testing
- `train_dataloader()`: Get training data loader
- `val_dataloader()`: Get validation data loader
- `test_dataloader()`: Get test data loader
- `build_input()`: Load and process data from files

### 3. EventTokenizer

Handles tokenization and encoding of event sequences.

```python
from easy_tpp.data.preprocess import EventTokenizer

tokenizer = EventTokenizer(tokenizer_specs)

# Tokenize a single sequence
encoded = tokenizer(
    time_seqs=[0.1, 0.3, 0.8, 1.2],
    time_delta_seqs=[0.1, 0.2, 0.5, 0.4],
    type_seqs=[0, 1, 0, 2],
    padding=True,
    truncation=True,
    max_length=50,
    return_tensors='pt'
)
```

### 4. TPPDataCollator

Dynamic padding and batching for variable-length sequences.

```python
from easy_tpp.data.preprocess import TPPDataCollator

data_collator = TPPDataCollator(
    tokenizer=tokenizer,
    padding=True,
    max_length=100,
    truncation=True,
    return_tensors="pt"
)

# Use with DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=data_collator
)
```

## Data Loading

### Supported Data Formats

#### JSON Format

```json
[
  {
    "time_since_start": [0.0, 0.123, 0.456, ...],
    "time_since_last_event": [0.0, 0.123, 0.333, ...],
    "type_event": [0, 1, 2, 0, 1, ...],
    "seq_len": 45,
    "dim_process": 3
  }
]
```

#### PKL Format

Pickle files containing preprocessed sequences with keys:

- `time_seqs`: Event timestamps
- `time_delta_seqs`: Inter-event intervals  
- `type_seqs`: Event type indices

### Loading Configuration

```python
data_config = DataConfig(
    # Data location
    data_dir='./data/',
    data_format='json',  # 'json' or 'pkl'
    
    # Data specifications
    tokenizer_specs={
        'num_event_types': 5,
        'max_seq_len': 100,
        'min_seq_len': 5,
        'padding_token_id': 0
    },
    
    # Loading specifications
    data_loading_specs={
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 8,
        'padding': 'longest',
        'truncation': True,
        'max_length': 128,
        'tensor_type': 'pt'  # 'pt', 'tf', or 'np'
    }
)
```

## Event Tokenization

### BatchEncoding

The tokenizer returns `BatchEncoding` objects containing:

```python
{
    'time_seqs': tensor([[...], [...]]),      # Padded timestamps
    'time_delta_seqs': tensor([[...], [...]]), # Padded intervals
    'type_seqs': tensor([[...], [...]]),      # Padded event types
    'attention_mask': tensor([[...], [...]]),  # Padding mask
    'seq_len': tensor([actual_lengths])       # Original lengths
}
```

### Padding Strategies

- `'longest'`: Pad to longest sequence in batch
- `'max_length'`: Pad to specified maximum length
- `False`: No padding (variable lengths)

### Truncation Strategies

- `True`: Truncate sequences exceeding max_length
- `False`: No truncation
- `'longest_first'`: Truncate longest sequences first

## Data Visualization

### Visualizer Class

Comprehensive visualization tools for data analysis and comparison.

```python
from easy_tpp.data.preprocess import Visualizer

visualizer = Visualizer(
    data_module=data_module,
    split='test',
    save_dir='./analysis/',
    dataset_size=5000
)
```

### Available Visualizations

#### 1. Event Type Distribution

```python
visualizer.plot_event_type_distribution()
```

Shows the frequency distribution of different event types.

#### 2. Inter-Event Time Analysis

```python
visualizer.plot_inter_event_time_distribution()
visualizer.plot_inter_event_time_histogram()
```

Analyzes the distribution of time intervals between events.

#### 3. Sequence Statistics

```python
visualizer.plot_sequence_length_distribution()
```

Shows distribution of sequence lengths in the dataset.

#### 4. Temporal Patterns

```python
visualizer.plot_event_intensity_over_time()
visualizer.plot_event_timeline()
```

Visualizes temporal patterns and event intensities.

### Comparison Mode

Compare two datasets side by side:

```python
visualizer = Visualizer(
    data_module=real_data_module,
    split='test',
    comparison_data_module=synthetic_data_module,
    comparison_split='test',
    save_dir='./comparison/'
)

visualizer.plot_comparison_statistics()
```

## Configuration

### Data Specifications

```python
tokenizer_specs = {
    'num_event_types': 10,       # Number of event types
    'max_seq_len': 200,          # Maximum sequence length
    'min_seq_len': 3,            # Minimum sequence length
    'pad_token_id': 0,           # Padding token ID
    'time_scale': 1.0,           # Time scaling factor
    'remove_empty_sequences': True
}
```

### Loading Specifications

```python
data_loading_specs = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True,
    'drop_last': False,
    'padding': True,
    'truncation': True,
    'max_length': 128,
    'tensor_type': 'pt'
}
```

## Examples

### Complete Pipeline Example

```python
from easy_tpp.data.preprocess import TPPDataModule, Visualizer
from easy_tpp.config_factory import DataConfig

def create_data_pipeline():
    # Configuration
    config = DataConfig(
        data_dir='./data/hawkes_multivariate/',
        data_format='json',
        tokenizer_specs={
            'num_event_types': 3,
            'max_seq_len': 100
        },
        data_loading_specs={
            'batch_size': 64,
            'shuffle': True,
            'num_workers': 8,
            'padding': 'longest',
            'truncation': True
        }
    )
    
    # Create data module
    data_module = TPPDataModule(config)
    data_module.setup()
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Analyze data
    visualizer = Visualizer(
        data_module=data_module,
        split='test',
        save_dir='./analysis/'
    )
    
    # Generate reports
    visualizer.plot_event_type_distribution()
    visualizer.plot_inter_event_time_distribution()
    visualizer.plot_sequence_length_distribution()
    
    return data_module, visualizer

# Run pipeline
data_module, visualizer = create_data_pipeline()
```

### Custom Data Loading

```python
def load_custom_data():
    # Manual data building
    data_module = TPPDataModule(config)
    
    # Load specific split
    test_data = data_module.build_input(
        source_dir='./data/test/',
        data_format='json',
        split='test'
    )
    
    # Create dataset
    test_dataset = TPPDataset(test_data)
    
    # Custom data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        collate_fn=data_module.tokenizer
    )
    
    return test_loader
```

### Data Validation

```python
def validate_data(data_module):
    """Validate loaded data for consistency."""
    
    # Setup data
    data_module.setup()
    
    # Check train data
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch size: {batch['type_seqs'].shape[0]}")
    print(f"Max sequence length: {batch['type_seqs'].shape[1]}")
    print(f"Event types range: {batch['type_seqs'].min()} - {batch['type_seqs'].max()}")
    
    # Validate shapes
    assert batch['time_seqs'].shape == batch['type_seqs'].shape
    assert batch['time_delta_seqs'].shape == batch['type_seqs'].shape
    assert batch['attention_mask'].shape == batch['type_seqs'].shape
    
    print("Data validation passed!")
    
    return True
```

## Integration with Training

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl

class TPPTrainer(pl.LightningModule):
    def __init__(self, model, data_module):
        super().__init__()
        self.model = model
        self.data_module = data_module
    
    def training_step(self, batch, batch_idx):
        # batch is already tokenized and collated
        outputs = self.model(
            time_seqs=batch['time_seqs'],
            time_delta_seqs=batch['time_delta_seqs'],
            type_seqs=batch['type_seqs'],
            attention_mask=batch['attention_mask']
        )
        return outputs.loss

# Usage
trainer = pl.Trainer()
trainer.fit(
    model=tpp_trainer,
    datamodule=data_module
)
```

## Best Practices

### 1. Memory Management

- Use appropriate `num_workers` for your system
- Enable `pin_memory=True` for GPU training
- Consider `drop_last=True` for consistent batch sizes

### 2. Data Preprocessing

- Validate data formats before training
- Use appropriate padding strategies for your model
- Monitor sequence length distributions

### 3. Performance Optimization

- Cache processed data when possible
- Use efficient data formats (PKL for large datasets)
- Profile data loading to identify bottlenecks

### 4. Data Quality

- Regularly visualize your data
- Check for data leakage between splits
- Validate event type ranges and distributions

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or number of workers
2. **Slow loading**: Use PKL format or increase num_workers
3. **Shape mismatches**: Check padding and truncation settings
4. **Missing data**: Verify file paths and data format

### Performance Tips

- Use PKL format for faster loading
- Optimize batch size for your hardware
- Use multiple workers for data loading
- Cache preprocessed data when possible

## Related Documentation

- [Data Generation](../generation/README.md): Synthetic data generation
- [Configuration Guide](../../config_factory/README.md): Configuration management
- [Training Examples](../../../examples/README.md): Complete training examples
- [Model Documentation](../../models/README.md): TPP model implementations
