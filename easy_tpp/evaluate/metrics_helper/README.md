# Metrics Helper Package

A modular metrics computation system for Temporal Point Process (TPP) models following SOLID principles.

## Overview

This package refactors the original monolithic `MetricsHelper` class into a modular, extensible system that follows SOLID principles:

- **Single Responsibility Principle (SRP)**: Each class has a single, well-defined responsibility
- **Open/Closed Principle (OCP)**: Open for extension, closed for modification
- **Liskov Substitution Principle (LSP)**: Implementations can be substituted without breaking functionality
- **Interface Segregation Principle (ISP)**: Focused interfaces that clients depend only on what they need
- **Dependency Inversion Principle (DIP)**: Depends on abstractions, not concretions

## Architecture

### Core Components

1. **MetricsHelper** (`main_metrics_helper.py`): Main orchestrator that delegates computation using the Strategy pattern
2. **PredictionMetricsComputer** (`prediction_metrics_computer.py`): Handles prediction-specific metrics
3. **SimulationMetricsComputer** (`simulation_metrics_computer.py`): Handles simulation-specific metrics
4. **Interfaces** (`interfaces.py`): Abstract interfaces for extensibility
5. **Shared Types** (`shared_types.py`): Common data structures and enums

### Design Patterns Used

- **Strategy Pattern**: MetricsHelper uses different strategies for prediction vs simulation metrics
- **Dependency Injection**: Computers can be injected for customization
- **Template Method**: Data extraction follows a consistent pattern

## Usage

For comprehensive usage examples and patterns, please refer to the existing test suite in the `tests/` folder of the main project.

## Available Metrics

### Prediction Metrics
- `time_rmse`: Root Mean Square Error for time predictions
- `time_mae`: Mean Absolute Error for time predictions
- `type_accuracy`: Classification accuracy for event types (multi-class only)
- `macro_f1score`: Macro-averaged F1 score (multi-class only)
- `micro_f1score`: Micro-averaged F1 score (multi-class only)
- `recall`: Recall score (multi-class only)
- `precision`: Precision score (multi-class only)
- `cross_entropy`: Cross-entropy loss (multi-class only)

### Simulation Metrics
- `wasserstein_1d`: Wasserstein distance between sequences
- `mmd_rbf_padded`: Maximum Mean Discrepancy with RBF kernel
- `mmd_wasserstein`: MMD with Wasserstein-based kernel

## Extension Points

### Creating Custom Metrics Computers

Implement the `MetricsComputerInterface`:

```python
from easy_tpp.evaluate.metrics_helper import MetricsComputerInterface

class CustomMetricsComputer(MetricsComputerInterface):
    def compute_metrics(self, batch, pred):
        # Implementation
        return {"custom_metric": 0.0}
    
    def get_available_metrics(self):
        return ["custom_metric"]
```

### Creating Custom Data Extractors

Implement the `DataExtractorInterface`:

```python
from easy_tpp.evaluate.metrics_helper import DataExtractorInterface

class CustomDataExtractor(DataExtractorInterface):
    def extract_values(self, batch, pred):
        # Custom extraction logic
        return extracted_data
```

## Migration from Legacy Code

The legacy `MetricsHelper` class is still available for backward compatibility but issues deprecation warnings. To migrate:

### Before (Legacy)
```python
from easy_tpp.evaluate import MetricsHelper, EvaluationMode

helper = MetricsHelper(num_event_types=5, mode=EvaluationMode.PREDICTION)
metrics = helper.compute_all_metrics(batch, pred)
```

### After (New)
```python
from easy_tpp.evaluate.metrics_helper import MetricsHelper, EvaluationMode

helper = MetricsHelper(num_event_types=5, mode=EvaluationMode.PREDICTION)
metrics = helper.compute_all_metrics(batch, pred)
```

## Benefits of the Refactored Design

1. **Modularity**: Each component has a clear, single responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Extensibility**: Easy to add new metrics without modifying existing code
4. **Maintainability**: Changes to one metric type don't affect others
5. **Reusability**: Components can be reused in different contexts
6. **Type Safety**: Clear interfaces and type annotations
7. **Performance**: No unnecessary computation for unused metrics

## File Structure

```
metrics_helper/
├── __init__.py                           # Package exports
├── interfaces.py                         # Abstract interfaces
├── shared_types.py                       # Common data structures
├── main_metrics_helper.py                # Main orchestrator
├── prediction_metrics_computer.py        # Prediction metrics
├── simulation_metrics_computer.py        # Simulation metrics
└── README.md                            # This file
```

This modular design makes the metrics system more maintainable, testable, and extensible while preserving backward compatibility.
