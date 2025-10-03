# Data Generation for Temporal Point Processes

This document provides a comprehensive guide for generating synthetic data using the new_ltpp framework's data generation capabilities.

## Overview

The data generation module provides tools for simulating various temporal point process models, including:

- **Hawkes Processes**: Self-exciting processes where past events increase the probability of future events
- **Self-Correcting Processes**: Processes where past events decrease the probability of future events
- **Multivariate Extensions**: Support for multiple event types and cross-dimensional influences

## Table of Contents

- [Quick Start](#quick-start)
- [Available Simulators](#available-simulators)
- [Data Generation Examples](#data-generation-examples)
- [Configuration and Parameters](#configuration-and-parameters)
- [Output Format](#output-format)
- [Utility Functions](#utility-functions)
- [Advanced Usage](#advanced-usage)

## Quick Start

### Basic Hawkes Process Simulation

```python
from new_ltpp.data_generation import HawkesSimulator

# Define Hawkes process parameters
generator = HawkesSimulator(
    mu=[0.2, 0.2],           # Base intensities
    alpha=[[0.4, 0], [0, 8]], # Excitation matrix
    beta=[[1, 0], [0, 20]],   # Decay matrix
    dim_process=2,            # Number of event types
    start_time=0,
    end_time=100
)

# Generate and save data
generator.generate_and_save(
    output_dir='./data/hawkes', 
    num_simulations=1000,
    splits={'train': 0.6, 'test': 0.2, 'dev': 0.2}
)
```

### Simple Data Generation with Utility Functions

```python
from new_ltpp.utils.gen_utils import generate_and_save_json

# Generate synthetic data using utility function
generate_and_save_json(
    n_nodes=3,
    end_time=100,
    baseline=1,
    adjacency=0.5,
    decay=0.1,
    max_seq_len=40,
    target_file='synthetic_data.json'
)
```

## Available Simulators

### 1. HawkesSimulator

Simulates multivariate Hawkes processes with exponential decay kernels.

**Mathematical Model:**

``` 
λᵢ(t) = μᵢ + Σⱼ Σₖ αᵢⱼ exp(-βᵢⱼ(t - tₖʲ))
```

**Parameters:**

- `mu`: Base intensities for each dimension (List[float])
- `alpha`: Excitation matrix (List[List[float]]) - how events in dimension j affect dimension i
- `beta`: Decay rates matrix (List[List[float]]) - decay speed of excitation effects
- `dim_process`: Number of event types (int)
- `start_time`, `end_time`: Simulation time window (float)

**Example:**

```python
from new_ltpp.data_generation import HawkesSimulator

simulator = HawkesSimulator(
    mu=[0.5, 0.3],                    # Base rates
    alpha=[[0.2, 0.1], [0.15, 0.25]], # Cross-excitation
    beta=[[1.0, 1.5], [2.0, 0.8]],    # Decay rates
    dim_process=2,
    start_time=0,
    end_time=50
)

# Single simulation
events = simulator.simulate()
print(f"Events in dimension 0: {events[0]}")
print(f"Events in dimension 1: {events[1]}")
```

### 2. SelfCorrecting

Simulates self-correcting processes where past events reduce future intensity.

**Mathematical Model:**

```
λᵢ(t) = μᵢ exp(xᵢ(t))
xᵢ(t) = μᵢ(t - t_last) - αᵢNᵢ(t)
```

**Parameters:**

- `mu`: Base rate parameter (float or List[float])
- `alpha`: Self-inhibition parameter (float or List[float])
- `dim_process`: Number of dimensions (int)

**Example:**

```python
from new_ltpp.data_generation import SelfCorrecting

simulator = SelfCorrecting(
    dim_process=2,
    mu=1.0,      # Base rate
    alpha=0.5,   # Self-correction strength
    start_time=0,
    end_time=100
)

events = simulator.simulate()
```

## Data Generation Examples

### Complete Pipeline Example

```python
from new_ltpp.data_generation import HawkesSimulator

def generate_hawkes_dataset():
    # Define parameters for a 3-dimensional Hawkes process
    params = {
        "mu": [0.2, 0.15, 0.1],
        "alpha": [
            [0.3, 0.1, 0.05],
            [0.1, 0.4, 0.1],
            [0.05, 0.1, 0.2]
        ],
        "beta": [
            [1.0, 1.5, 2.0],
            [1.2, 0.8, 1.8],
            [2.0, 1.0, 1.5]
        ]
    }
    
    # Create simulator
    generator = HawkesSimulator(
        mu=params["mu"],
        alpha=params["alpha"],
        beta=params["beta"],
        dim_process=3,
        start_time=0,
        end_time=200
    )
    
    # Generate data with custom splits
    generator.generate_and_save(
        output_dir='./data/hawkes_3d',
        num_simulations=2000,
        splits={'train': 0.7, 'test': 0.15, 'dev': 0.15}
    )
    
    return generator

# Run the generation
simulator = generate_hawkes_dataset()
```

### Intensity Visualization

```python
# Generate intensity plots
intensities, time_points, events = simulator.intensity_graph(
    precision=1000,
    plot=True,
    save_plot=True,
    save_data=True,
    save_dir='./plots/'
)
```

## Configuration and Parameters

### Common Parameters for All Simulators

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim_process` | int | Number of event types/dimensions |
| `start_time` | float | Start time of simulation window |
| `end_time` | float | End time of simulation window |
| `seed` | int (optional) | Random seed for reproducibility |

### Hawkes-Specific Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `mu` | List[float] | [dim] | Base intensities |
| `alpha` | List[List[float]] | [dim, dim] | Excitation matrix |
| `beta` | List[List[float]] | [dim, dim] | Decay rate matrix |

### Data Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_simulations` | int | 1000 | Number of sequences to generate |
| `output_dir` | str | 'data' | Output directory |
| `splits` | Dict[str, float] | {'train': 0.6, 'test': 0.2, 'dev': 0.2} | Data splits |

## Output Format

### Directory Structure

``` bash
output_dir/
├── train.json          # Training data
├── test.json           # Test data  
├── dev.json            # Development data
└── metadata.json       # Simulation metadata
```

### Data Format (JSON)

Each JSON file contains a list of sequences:

```json
[
  {
    "dim_process": 3,
    "seq_len": 45,
    "seq_idx": 0,
    "time_since_start": [0.0, 0.123, 0.456, ...],
    "time_since_last_event": [0.0, 0.123, 0.333, ...],
    "type_event": [0, 1, 2, 0, 1, ...]
  },
  ...
]
```

### Metadata Format

```json
{
  "simulation_info": {
    "num_simulations": 1000,
    "dimension": 3,
    "time_interval": [0, 200],
    "simulator_type": "HawkesSimulator"
  },
  "split_info": {
    "train_size": 600,
    "test_size": 200,
    "dev_size": 200
  },
  "hawkes_parameters": {
    "mu": [0.2, 0.15, 0.1],
    "alpha": [[0.3, 0.1, 0.05], ...],
    "beta": [[1.0, 1.5, 2.0], ...]
  }
}
```

## Utility Functions

### generate_and_save_json()

Simple utility for quick data generation:

```python
from new_ltpp.utils.gen_utils import generate_and_save_json

generate_and_save_json(
    n_nodes=3,           # Number of dimensions
    end_time=100,        # Simulation end time
    baseline=0.5,        # Base intensity
    adjacency=0.3,       # Cross-excitation strength
    decay=1.0,           # Decay rate
    max_seq_len=50,      # Maximum sequence length
    target_file='data.json'
)
```

### Direct Data Generation

```python
from new_ltpp.utils.gen_utils import generate_synthetic_data

# Generate raw event data
events = generate_synthetic_data(
    n_nodes=3,
    end_time=1000,
    baseline=0.1,
    adjacency=0.5,
    decay=1.0
)

# events is a list of lists containing event dictionaries
# events[i] contains all events for dimension i
```

## Advanced Usage

### Custom Simulation Loop

```python
from new_ltpp.data_generation import HawkesSimulator

simulator = HawkesSimulator(
    mu=[0.3, 0.2],
    alpha=[[0.1, 0.2], [0.15, 0.1]],
    beta=[[1.0, 1.5], [1.2, 0.8]],
    dim_process=2,
    start_time=0,
    end_time=50
)

# Generate multiple realizations
all_simulations = []
for i in range(100):
    events = simulator.simulate()
    all_simulations.append(events)

# Custom formatting
formatted_data = simulator.format_multivariate_simulations(
    all_simulations,
    dim_process=2,
    start_time=0
)
```

### Batch Processing with Progress Tracking

```python
# Generate large datasets with progress tracking
large_dataset = simulator.bulk_simulate(num_simulations=5000)

# Custom train/test splits
train_data, test_data, dev_data = simulator.split_data(
    large_dataset,
    train_ratio=0.8,
    test_ratio=0.1,
    dev_ratio=0.1
)
```

### Intensity Analysis

```python
# Analyze theoretical intensities
intensities, time_points, events = simulator.intensity_graph(
    precision=500,          # Number of time points
    plot=False,            # Don't display plot
    save_plot=True,        # Save plot to file
    save_data=True,        # Save intensity data
    save_dir='./analysis/'
)

# intensities: [time_points, dim_process] array
# time_points: [time_points] array  
# events: dict with events by dimension
```

## Tips and Best Practices

### 1. Parameter Selection

- Start with small `alpha` values (< 0.5) to avoid explosive behavior
- Use `beta` values > 1 for realistic decay
- Balance `mu` values based on desired event rates

### 2. Simulation Length

- Use longer simulation windows for more stable statistics
- Consider computational cost vs. data quality trade-offs

### 3. Data Quality

- Check generated sequences for reasonable event rates
- Verify that intensity patterns match expectations
- Use visualization tools to inspect generated data

### 4. Reproducibility

- Always set random seeds for reproducible results
- Save metadata with generation parameters
- Version control your generation scripts

## Troubleshooting

### Common Issues

1. **Empty sequences**: Increase `end_time` or `mu` values
2. **Explosive behavior**: Reduce `alpha` values or increase `beta`
3. **Memory issues**: Reduce `num_simulations` or use batch processing
4. **Slow generation**: Optimize parameters or reduce simulation complexity

### Performance Tips

- Use appropriate time windows (not too large)
- Balance number of simulations vs. sequence length
- Consider parallel processing for large datasets

## Examples Directory

Check the `examples/` and `main/data_gen/` directories for additional examples:

- `examples/gen_synthetic_data.py`: Basic data generation
- `main/data_gen/run_gen.py`: Complete pipeline example

## Related Documentation

- [Main README](README.md): General project overview
- [Model Documentation](docs/): TPP model details
- [Configuration Guide](docs/): Parameter configuration
- [Dataset Documentation](docs/user_guide/dataset.rst): Data formats and preprocessing
