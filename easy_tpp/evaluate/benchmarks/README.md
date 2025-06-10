# TPP Benchmarks Module

This module provides baseline benchmark implementations for evaluating Temporal Point Process (TPP) models. The benchmarks offer simple prediction strategies that serve as performance baselines for more complex models.

## Overview

The benchmark module has been refactored to support flexible evaluation modes, allowing you to evaluate either time predictions, type predictions, or both independently. This enables more targeted performance analysis and supports different model architectures that may specialize in either time or type prediction.

## Architecture

### BenchmarkMode

The `BenchmarkMode` enum defines three evaluation modes:

- **`TIME_ONLY`**: Evaluate only time-based metrics (RMSE, MAE, etc.)
- **`TYPE_ONLY`**: Evaluate only type-based metrics (accuracy, F1-score, etc.)
- **`BOTH`**: Evaluate all metrics (legacy behavior)

### Base Classes

#### `BaseBenchmark`

Abstract base class that all benchmarks inherit from. Provides:

- Data loading and preprocessing
- Metrics computation using the `MetricsHelper`
- Result aggregation and saving
- Support for different evaluation modes

#### Key Methods

- `_create_predictions(batch)`: Legacy method returning (time_pred, type_pred) tuple
- `_create_time_predictions(batch)`: Specialized method returning only time predictions
- `_create_type_predictions(batch)`: Specialized method returning only type predictions
- `evaluate()`: Main evaluation method that orchestrates the benchmark process

## Available Benchmarks

### 1. LastMarkBenchmark

**Default Mode**: `TYPE_ONLY`

Predicts the next event type using the previous event type (lag-1 strategy). This is a simple baseline for type prediction tasks.

```python
from easy_tpp.evaluate.benchmarks import LastMarkBenchmark, BenchmarkMode

# Use default TYPE_ONLY mode
benchmark = LastMarkBenchmark(data_config, "experiment_id")
```

### 2. MeanInterTimeBenchmark

**Default Mode**: `TIME_ONLY`

Predicts all inter-event times using the mean inter-time computed from training data. This is a simple baseline for time prediction tasks.

```python
from easy_tpp.evaluate.benchmarks import MeanInterTimeBenchmark

# Use default TIME_ONLY mode
benchmark = MeanInterTimeBenchmark(data_config, "experiment_id")
```

### 3. MarkDistributionBenchmark

**Default Mode**: `TYPE_ONLY`

Predicts event types by sampling from the empirical distribution of event types in the training data.

```python
from easy_tpp.evaluate.benchmarks import MarkDistributionBenchmark

# Use default TYPE_ONLY mode
benchmark = MarkDistributionBenchmark(data_config, "experiment_id")
```

### 4. InterTimeDistributionBenchmark

**Default Mode**: `TIME_ONLY`

Predicts inter-event times by sampling from a histogram-based approximation of the inter-time distribution in the training data.

```python
from easy_tpp.evaluate.benchmarks import InterTimeDistributionBenchmark

# Use default TIME_ONLY mode with 50 bins
benchmark = InterTimeDistributionBenchmark(data_config, "experiment_id")

# Use custom number of bins
benchmark = InterTimeDistributionBenchmark(data_config, "experiment_id", num_bins=100)
```

## Usage Examples

### Basic Usage

```python
import yaml
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.evaluate.benchmarks import LastMarkBenchmark

# Load configuration
with open("config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
data_config = DataConfig.from_dict(config_dict["data_config"])

# Run benchmark
benchmark = LastMarkBenchmark(data_config, "my_experiment")
results = benchmark.evaluate()

print(f"Type Accuracy: {results['metrics']['type_accuracy_mean']:.4f}")
```

### Custom Evaluation Modes

```python
from easy_tpp.evaluate.benchmarks import MeanInterTimeBenchmark, BenchmarkMode

# Evaluate only time metrics (default for this benchmark)
time_benchmark = MeanInterTimeBenchmark(data_config, "exp1")

### Running Multiple Benchmarks

```python
from easy_tpp.evaluate.benchmarks import (
    LastMarkBenchmark, MeanInterTimeBenchmark, 
    MarkDistributionBenchmark, InterTimeDistributionBenchmark
)

benchmarks = [
    LastMarkBenchmark(data_config, "exp1"),  # TYPE_ONLY
    MeanInterTimeBenchmark(data_config, "exp2"),  # TIME_ONLY
    MarkDistributionBenchmark(data_config, "exp3"),  # TYPE_ONLY
    InterTimeDistributionBenchmark(data_config, "exp4")  # TIME_ONLY
]

results = {}
for benchmark in benchmarks:
    results[benchmark.benchmark_name] = benchmark.evaluate()
```

### Command Line Usage

Each benchmark can be run from the command line:

```bash
# Run LastMarkBenchmark
python -m easy_tpp.evaluate.benchmarks.last_mark_bench \
    --config_path config.yaml \
    --experiment_id my_experiment \
    --save_dir ./results

# Run MeanInterTimeBenchmark
python -m easy_tpp.evaluate.benchmarks.mean_bench \
    --config_path config.yaml \
    --experiment_id my_experiment \
    --save_dir ./results

# Run InterTimeDistributionBenchmark with custom bins
python -m easy_tpp.evaluate.benchmarks.sample_distrib_intertime_bench \
    --config_path config.yaml \
    --experiment_id my_experiment \
    --num_bins 100 \
    --save_dir ./results
```

## Creating Custom Benchmarks

To create a custom benchmark, inherit from `BaseBenchmark` and implement the required methods:

```python
from easy_tpp.evaluate.benchmarks import BaseBenchmark, BenchmarkMode

class CustomBenchmark(BaseBenchmark):
    def __init__(self, data_config, experiment_id, save_dir=None, 
                 benchmark_mode=BenchmarkMode.BOTH):
        super().__init__(data_config, experiment_id, save_dir, benchmark_mode)
    
    @property
    def benchmark_name(self) -> str:
        return "custom_benchmark"
    
    def _prepare_benchmark(self) -> None:
        # Compute any necessary statistics from training data
        pass
    
    def _create_predictions(self, batch) -> tuple:
        # Legacy method - return (time_pred, type_pred)
        time_pred = # ... your time prediction logic
        type_pred = # ... your type prediction logic
        return time_pred, type_pred
    
    def _create_time_predictions(self, batch) -> torch.Tensor:
        # Specialized method for time predictions only
        return # ... your time prediction logic
    
    def _create_type_predictions(self, batch) -> torch.Tensor:
        # Specialized method for type predictions only
        return # ... your type prediction logic
```

## Results Format

Benchmark results are saved as JSON files with the following structure:

```json
{
  "benchmark_name": "lag1_mark_benchmark",
  "dataset_name": "experiment_id",
  "num_event_types": 5,
  "metrics": {
    "type_accuracy_mean": 0.6234,
    "type_accuracy_std": 0.0123,
    "type_accuracy_min": 0.6100,
    "type_accuracy_max": 0.6350,
    "macro_f1score_mean": 0.5123,
    "macro_f1score_std": 0.0089,
    // ... other metrics
  },
  "num_batches_evaluated": 150,
  // Custom benchmark-specific information
  "strategy": "lag1_mark_prediction"
}
```

## Metrics Computed

### Time-based Metrics (TIME_ONLY or BOTH modes)
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error
- **Time Accuracy**: Accuracy within tolerance

### Type-based Metrics (TYPE_ONLY or BOTH modes)
- **Type Accuracy**: Classification accuracy
- **Macro F1-Score**: F1-score averaged across all event types
- **Micro F1-Score**: F1-score computed globally
- **Per-class Precision/Recall**: Individual metrics for each event type

## Migration from Legacy Code

The refactored architecture maintains backward compatibility. Existing code will continue to work:

```python
# Old style - still works
benchmark = LastMarkBenchmark(data_config, "exp1")
results = benchmark.evaluate()  # Will use TYPE_ONLY mode by default
```

## Performance Considerations

- **TYPE_ONLY** and **TIME_ONLY** modes are more efficient as they compute fewer metrics
- Benchmarks automatically set appropriate default modes based on their focus
- Use **BOTH** mode only when you need comprehensive evaluation
- Large datasets benefit from the specialized modes due to reduced computation

## Dependencies

- PyTorch
- NumPy
- PyYAML
- easy_tpp core modules (MetricsHelper, DataConfig, etc.)

## Contributing

When adding new benchmarks:

1. Inherit from `BaseBenchmark`
2. Set an appropriate default `benchmark_mode`
3. Implement all three prediction methods
4. Add comprehensive tests
5. Update this README with usage examples
