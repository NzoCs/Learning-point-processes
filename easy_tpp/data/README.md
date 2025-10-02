# Data Management for Temporal Point Processes

This directory contains the complete data pipeline for temporal point process modeling in the EasyTPP framework. It provides comprehensive tools for both generating synthetic data and preprocessing real-world datasets.

## Overview

The data module consists of two main components:

- **[Generation](generation/)**: Synthetic data generation for various temporal point process models
- **[Preprocess](preprocess/)**: Data loading, tokenization, and preprocessing pipeline

## Directory Structure

``` bash
data/
├── generation/           # Synthetic data generation
│   ├── base_simulator.py    # Abstract base simulator class
│   ├── hawkes.py           # Hawkes process simulation
│   ├── self_correcting.py  # Self-correcting process simulation
│   └── README.md           # Generation documentation
└── preprocess/          # Data preprocessing pipeline
    ├── dataset.py          # TPP dataset implementation
    ├── data_loader.py      # PyTorch Lightning data module
    ├── data_collator.py    # Data collation utilities
    ├── event_tokenizer.py  # Event tokenization
    ├── visualizer.py       # Data visualization tools
    └── README.md           # Preprocessing documentation
```

## Quick Start

### Generate Synthetic Data

```python
from easy_tpp.data.generation import HawkesSimulator

# Create Hawkes process simulator
simulator = HawkesSimulator(
    mu=[0.2, 0.2],                    # Base intensities
    alpha=[[0.4, 0.1], [0.15, 0.25]], # Cross-excitation
    beta=[[1.0, 1.5], [2.0, 0.8]],    # Decay rates
    dim_process=2,
    start_time=100,
    end_time=200
)

# Generate and save dataset
simulator.generate_and_save(
    output_dir='./data/synthetic_hawkes',
    num_simulations=1000,
    splits={'train': 0.6, 'test': 0.2, 'dev': 0.2}
)
```

### Load and Preprocess Data

```python
from easy_tpp.data.preprocess import TPPDataModule
from easy_tpp.config_factory import DataConfig

# Configure data loading
data_config = DataConfig(
    data_dir='./data/synthetic_hawkes',
    data_format='json',
    tokenizer_specs={
        'num_event_types': 2,
        'max_seq_len': 100
    },
    data_loading_specs={
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4
    }
)

# Create data module
data_module = TPPDataModule(data_config)
data_module.setup()

# Get data loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

## Complete Pipeline Example

### End-to-End Workflow

```python
from easy_tpp.data.generation import HawkesSimulator
from easy_tpp.data.preprocess import TPPDataModule, Visualizer
from easy_tpp.config_factory import DataConfig

def create_complete_data_pipeline():
    """Complete example: generate data, preprocess, and visualize."""
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic data...")
    simulator = HawkesSimulator(
        mu=[0.3, 0.2, 0.15],
        alpha=[
            [0.2, 0.1, 0.05],
            [0.1, 0.3, 0.1],
            [0.05, 0.1, 0.25]
        ],
        beta=[
            [1.0, 1.5, 2.0],
            [1.2, 0.8, 1.8],
            [2.0, 1.0, 1.5]
        ],
        dim_process=3,
        start_time=100,
        end_time=300
    )
    
    # Generate dataset
    simulator.generate_and_save(
        output_dir='./data/hawkes_3d',
        num_simulations=2000,
        splits={'train': 0.7, 'test': 0.15, 'dev': 0.15}
    )
    
    # Step 2: Configure data preprocessing
    print("Step 2: Setting up data preprocessing...")
    data_config = DataConfig(
        data_dir='./data/hawkes_3d',
        data_format='json',
        tokenizer_specs={
            'num_event_types': 3,
            'max_seq_len': 150,
            'min_seq_len': 5
        },
        data_loading_specs={
            'batch_size': 64,
            'shuffle': True,
            'num_workers': 8,
            'padding': 'longest',
            'truncation': True,
            'max_length': 200
        }
    )
    
    # Create data module
    data_module = TPPDataModule(data_config)
    data_module.setup()
    
    # Step 3: Analyze and visualize data
    print("Step 3: Analyzing generated data...")
    visualizer = Visualizer(
        data_module=data_module,
        split='test',
        save_dir='./analysis/hawkes_3d/',
        dataset_size=1000
    )
    
    # Generate comprehensive analysis
    visualizer.plot_event_type_distribution()
    visualizer.plot_inter_event_time_distribution()
    visualizer.plot_sequence_length_distribution()
    
    print("Pipeline completed successfully!")
    return simulator, data_module, visualizer

# Run the complete pipeline
simulator, data_module, visualizer = create_complete_data_pipeline()
```

## Data Flow Architecture

``` bash
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Generation    │───▶│   Raw Data       │───▶│  Preprocessing  │
│                 │    │                  │    │                 │
│ • HawkesSimulator│    │ • JSON files     │    │ • TPPDataModule │
│ • SelfCorrecting│    │ • PKL files      │    │ • EventTokenizer│
│ • BaseSimulator │    │ • Metadata       │    │ • TPPDataCollator│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Visualization │◀───│  Processed Data  │───▶│   Model Input   │
│                 │    │                  │    │                 │
│ • Visualizer    │    │ • Tokenized      │    │ • Batched       │
│ • Statistics    │    │ • Padded         │    │ • Masked        │
│ • Comparisons   │    │ • Collated       │    │ • Ready for GPU │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Features

### Data Generation

- **Multiple TPP Models**: Hawkes processes, self-correcting processes
- **Multivariate Support**: Cross-dimensional event interactions
- **Flexible Configuration**: Customizable parameters and time windows
- **Automatic Formatting**: Direct output in training-ready format
- **Intensity Analysis**: Built-in visualization and analysis tools

### Data Preprocessing  

- **Multiple Formats**: JSON, PKL support with automatic detection
- **PyTorch Lightning**: Seamless integration with modern training pipelines
- **Dynamic Batching**: Efficient padding and collation for variable-length sequences
- **Tokenization**: Comprehensive event encoding with attention masks
- **Visualization**: Rich analysis and comparison tools

### Integration Benefits

- **Consistent Format**: Generated data is immediately compatible with preprocessing
- **End-to-End Pipeline**: From simulation to model training
- **Quality Assurance**: Built-in validation and analysis tools
- **Scalability**: Efficient handling of large datasets

## Configuration Management

### Unified Configuration

```python
from easy_tpp.config_factory import Config

# Complete configuration for both generation and preprocessing
config = Config(
    # Generation settings
    generation={
        'simulator_type': 'HawkesSimulator',
        'num_simulations': 5000,
        'mu': [0.2, 0.15, 0.1],
        'alpha': [[0.3, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.2]],
        'beta': [[1.0, 1.5, 2.0], [1.2, 0.8, 1.8], [2.0, 1.0, 1.5]],
        'dim_process': 3,
        'time_window': [100, 300],
        'output_dir': './data/generated'
    },
    
    # Data processing settings
    data={
        'data_dir': './data/generated',
        'data_format': 'json',
        'tokenizer_specs': {
            'num_event_types': 3,
            'max_seq_len': 200
        },
        'data_loading_specs': {
            'batch_size': 64,
            'shuffle': True,
            'num_workers': 8
        }
    }
)
```

## Data Quality Assurance

### Validation Pipeline

```python
def validate_data_pipeline(data_module, expected_event_types=None):
    """Comprehensive data validation."""
    
    # Setup and load data
    data_module.setup()
    
    # Validate train data
    train_loader = data_module.train_dataloader()
    train_batch = next(iter(train_loader))
    
    # Basic shape validation
    batch_size, seq_len = train_batch['type_seqs'].shape
    print(f"✓ Batch shape: {batch_size} x {seq_len}")
    
    # Event type validation
    unique_types = train_batch['type_seqs'].unique().tolist()
    if expected_event_types:
        assert set(unique_types).issubset(set(range(expected_event_types)))
        print(f"✓ Event types valid: {unique_types}")
    
    # Temporal consistency
    time_seqs = train_batch['time_seqs']
    time_deltas = train_batch['time_delta_seqs']
    
    # Check non-negative times
    assert (time_seqs >= 0).all(), "Negative timestamps found"
    assert (time_deltas >= 0).all(), "Negative time deltas found"
    print("✓ Temporal consistency validated")
    
    # Attention mask validation
    attention_mask = train_batch['attention_mask']
    assert attention_mask.dtype == torch.bool or attention_mask.dtype == torch.long
    print("✓ Attention masks validated")
    
    return True

# Usage
validate_data_pipeline(data_module, expected_event_types=3)
```

## Performance Optimization

### Memory Efficiency

```python
# Optimize for large datasets
data_config = DataConfig(
    data_loading_specs={
        'batch_size': 32,          # Adjust based on GPU memory
        'num_workers': 8,          # CPU core count
        'pin_memory': True,        # GPU acceleration
        'persistent_workers': True, # Reduce worker startup overhead
        'prefetch_factor': 2       # Background data loading
    }
)
```

### Processing Speed

```python
# Use PKL format for large datasets
def convert_json_to_pkl(json_dir, pkl_dir):
    """Convert JSON data to PKL for faster loading."""
    
    from easy_tpp.utils import save_pickle, load_json
    import os
    
    os.makedirs(pkl_dir, exist_ok=True)
    
    for split in ['train', 'dev', 'test']:
        json_file = os.path.join(json_dir, f'{split}.json')
        pkl_file = os.path.join(pkl_dir, f'{split}.pkl')
        
        if os.path.exists(json_file):
            data = load_json(json_file)
            save_pickle(data, pkl_file)
            print(f"Converted {split}: {json_file} -> {pkl_file}")
```

## Common Workflows

### 1. Synthetic Data Experiment

```python
def run_synthetic_experiment():
    """Run experiment with synthetic data."""
    
    # Generate multiple datasets with different parameters
    configs = [
        {'mu': [0.1, 0.1], 'alpha': [[0.2, 0.1], [0.1, 0.3]]},
        {'mu': [0.2, 0.2], 'alpha': [[0.4, 0.2], [0.2, 0.5]]},
        {'mu': [0.3, 0.3], 'alpha': [[0.6, 0.3], [0.3, 0.7]]}
    ]
    
    for i, config in enumerate(configs):
        # Generate data
        simulator = HawkesSimulator(
            **config,
            beta=[[1.0, 1.0], [1.0, 1.0]],
            dim_process=2,
            start_time=0,
            end_time=100
        )
        
        output_dir = f'./experiments/config_{i}'
        simulator.generate_and_save(output_dir, num_simulations=1000)
        
        # Analyze data
        data_module = create_data_module(output_dir)
        visualizer = Visualizer(data_module, save_dir=output_dir)
        visualizer.plot_event_type_distribution()
```

### 2. Real Data Processing

```python
def process_real_data(raw_data_path, processed_data_path):
    """Process real-world temporal data."""
    
    # Custom preprocessing for real data
    # (Implementation depends on data format)
    
    # Create data module
    data_config = DataConfig(
        data_dir=processed_data_path,
        data_format='json',
        tokenizer_specs={'num_event_types': 'auto_detect'},
        data_loading_specs={'batch_size': 64}
    )
    
    data_module = TPPDataModule(data_config)
    return data_module
```

### 3. Data Comparison

```python
def compare_datasets(real_data_module, synthetic_data_module):
    """Compare real and synthetic datasets."""
    
    visualizer = Visualizer(
        data_module=real_data_module,
        split='test',
        comparison_data_module=synthetic_data_module,
        comparison_split='test',
        save_dir='./comparison_analysis/'
    )
    
    # Generate comparison plots
    visualizer.plot_comparison_statistics()
    visualizer.plot_event_type_distribution()
    visualizer.plot_inter_event_time_distribution()
```

## Best Practices

### 1. Data Generation

- Start with simple parameters and gradually increase complexity
- Validate generated data matches theoretical expectations
- Use multiple random seeds for robust experiments
- Save generation parameters with the data

### 2. Data Preprocessing

- Always validate data after loading
- Use appropriate padding strategies for your model
- Monitor memory usage with large datasets
- Profile data loading performance

### 3. Pipeline Integration

- Keep consistent data formats across pipeline stages
- Document parameter choices and data transformations
- Use version control for data and configurations
- Implement comprehensive testing for data quality

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|--------|----------|
| Memory errors | Large batch size or datasets | Reduce batch size, use more workers |
| Slow data loading | Inefficient format or I/O | Use PKL format, optimize num_workers |
| Shape mismatches | Inconsistent padding/truncation | Check tokenization configuration |
| Missing data files | Incorrect paths | Verify file paths and formats |
| GPU memory errors | Large sequences or batches | Reduce max_length or batch_size |

### Performance Debugging

```python
def profile_data_loading(data_module):
    """Profile data loading performance."""
    
    import time
    
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    # Time first batch (includes initialization)
    start_time = time.time()
    first_batch = next(iter(train_loader))
    first_batch_time = time.time() - start_time
    
    # Time subsequent batches
    batch_times = []
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Sample 10 batches
            break
        start_time = time.time()
        _ = batch  # Just access the batch
        batch_times.append(time.time() - start_time)
    
    print(f"First batch time: {first_batch_time:.3f}s")
    print(f"Average batch time: {np.mean(batch_times):.3f}s")
    print(f"Batch size: {first_batch['type_seqs'].shape[0]}")
    
    return np.mean(batch_times)
```

## Related Documentation

- [Generation Module](generation/README.md): Detailed synthetic data generation guide
- [Preprocessing Module](preprocess/README.md): Comprehensive preprocessing documentation
- [Configuration Guide](../config_factory/README.md): Configuration management
- [Model Documentation](../models/README.md): TPP model implementations
- [Examples](../../examples/README.md): Complete usage examples

## Contributing

When contributing to the data module:

1. Ensure new simulators inherit from `BaseSimulator`
2. Add comprehensive tests for new functionality
3. Update documentation and examples
4. Validate data format compatibility
5. Consider performance implications

For detailed contribution guidelines, see the main project documentation.
