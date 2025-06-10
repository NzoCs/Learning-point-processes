# Metrics Helper Package

A modular metrics computation system for Temporal Point Process (TPP) with specialized handling for time and type metrics.

## Architecture

### Core Components

1. **MetricsHelper** (`main_metrics_helper.py`): Main orchestrator that delegates computation using the Strategy pattern
2. **PredictionMetricsComputer** (`prediction_metrics_computer.py`): Handles prediction-specific metrics with separate time and type computation
3. **SimulationMetricsComputer** (`simulation_metrics_computer.py`): Handles simulation-specific metrics
4. **Interfaces** (`interfaces.py`): Abstract interfaces for extensibility, including specialized extractors
5. **Shared Types** (`shared_types.py`): Common data structures, enums, and type-safe containers

### Specialized Data Extractors

The architecture now includes specialized extractors for granular data handling:

**Prediction Extractors:**
- **TimeDataExtractor**: Extracts time-related values (`TimeValues` container)
- **TypeDataExtractor**: Extracts type-related values (`TypeValues` container)  
- **PredictionDataExtractor**: Legacy-compatible extractor (`MaskedValues` container)

**Simulation Extractors:**
- **SimulationTimeDataExtractor**: Extracts simulation time-related values (`SimulationTimeValues` container)
- **SimulationTypeDataExtractor**: Extracts simulation type-related values (`SimulationTypeValues` container)
- **SimulationDataExtractor**: Legacy-compatible simulation extractor

### Type-Safe Data Containers

**Prediction Containers:**
- **`TimeValues`**: Container for time-related predictions and ground truth
- **`TypeValues`**: Container for type-related predictions and ground truth
- **`MaskedValues`**: Combined container for backward compatibility

**Simulation Containers:**
- **`SimulationTimeValues`**: Container for simulation time sequences and masks
- **`SimulationTypeValues`**: Container for simulation type sequences and masks  
- **`SimulationValues`**: Combined container for backward compatibility

### Design Patterns Used

- **Strategy Pattern**: MetricsHelper uses different strategies for prediction vs simulation metrics
- **Dependency Injection**: Computers and extractors can be injected for customization
- **Interface Segregation**: Separate interfaces for time, type, and general data extraction
- **Template Method**: Data extraction follows a consistent pattern with specialized implementations

## Usage

### Basic Usage

```python
from easy_tpp.evaluate.metrics_helper import MetricsHelper, EvaluationMode

# Initialize for prediction metrics
helper = MetricsHelper(
    num_event_types=5,
    mode=EvaluationMode.PREDICTION
)

# Compute all metrics
all_metrics = helper.compute_all_metrics(batch, pred)

# Compute only time-related metrics
time_metrics = helper.compute_all_time_metrics(batch, pred)

# Compute only type-related metrics  
type_metrics = helper.compute_all_type_metrics(batch, pred)
```

### Selective Metric Computation

```python
from easy_tpp.evaluate.metrics_helper import MetricsHelper, PredictionMetrics

# Initialize with selected metrics only
helper = MetricsHelper(
    num_event_types=5,
    mode=EvaluationMode.PREDICTION,
    selected_prediction_metrics=[
        PredictionMetrics.TIME_RMSE,
        PredictionMetrics.TIME_MAE,
        PredictionMetrics.TYPE_ACCURACY
    ]
)

# Only selected metrics will be computed
metrics = helper.compute_all_metrics(batch, pred)
```

### Mode Switching

```python
# Switch between prediction and simulation modes
helper.set_mode(EvaluationMode.SIMULATION)
sim_metrics = helper.compute_all_metrics(batch, pred)

# Compute only time-related simulation metrics
sim_time_metrics = helper.compute_all_simulation_time_metrics(batch, pred)

# Compute only type-related simulation metrics  
sim_type_metrics = helper.compute_all_simulation_type_metrics(batch, pred)

helper.set_mode(EvaluationMode.PREDICTION)
pred_metrics = helper.compute_all_metrics(batch, pred)
```

For comprehensive usage examples and patterns, please refer to the existing test suite in the `tests/` folder of the main project.

## Available Metrics

### Prediction Metrics

#### Time Metrics
- `time_rmse`: Root Mean Square Error for time predictions
- `time_mae`: Mean Absolute Error for time predictions

#### Type Metrics (Multi-class only)
- `type_accuracy`: Classification accuracy for event types
- `macro_f1score`: Macro-averaged F1 score
- `micro_f1score`: Micro-averaged F1 score  
- `recall`: Recall score
- `precision`: Precision score
- `cross_entropy`: Cross-entropy loss
- `confusion_matrix`: Confusion matrix for detailed analysis

### Simulation Metrics

#### Time-based Metrics
- `wasserstein_1d`: Wasserstein distance between temporal sequences
- `mmd_rbf_padded`: Maximum Mean Discrepancy with RBF kernel on padded sequences
- `mmd_wasserstein`: MMD with Wasserstein-based kernel for temporal distributions

#### Type-based Metrics
- Currently, simulation metrics focus primarily on temporal patterns
- Future extensions may include type sequence similarity metrics
- Type distribution comparison metrics could be added

### Metric Selection

Metrics can be selected using enum values or strings:

```python
# Using enum values (recommended)
selected_metrics = [
    PredictionMetrics.TIME_RMSE,
    PredictionMetrics.TYPE_ACCURACY
]

# Using strings
selected_metrics = ['time_rmse', 'type_accuracy']
```

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

#### General Data Extractor
```python
from easy_tpp.evaluate.metrics_helper import DataExtractorInterface

class CustomDataExtractor(DataExtractorInterface):
    def extract_values(self, batch, pred):
        # Custom extraction logic
        return extracted_data
```

#### Specialized Time Extractor
```python
from easy_tpp.evaluate.metrics_helper import TimeExtractorInterface, TimeValues

class CustomTimeExtractor(TimeExtractorInterface):
    def extract_time_values(self, batch, pred) -> TimeValues:
        # Extract time-specific data
        return TimeValues(true_times, pred_times)
```

#### Specialized Type Extractor
```python
from easy_tpp.evaluate.metrics_helper import TypeExtractorInterface, TypeValues

class CustomTypeExtractor(TypeExtractorInterface):
    def extract_type_values(self, batch, pred) -> TypeValues:
        # Extract type-specific data
        return TypeValues(true_types, pred_types)
```

#### Specialized Simulation Time Extractor
```python
from easy_tpp.evaluate.metrics_helper import SimulationTimeExtractorInterface, SimulationTimeValues

class CustomSimulationTimeExtractor(SimulationTimeExtractorInterface):
    def extract_simulation_time_values(self, batch, pred) -> SimulationTimeValues:
        # Extract simulation time-specific data
        return SimulationTimeValues(
            true_time_seqs, true_time_delta_seqs,
            sim_time_seqs, sim_time_delta_seqs, sim_mask
        )
```

#### Specialized Simulation Type Extractor
```python
from easy_tpp.evaluate.metrics_helper import SimulationTypeExtractorInterface, SimulationTypeValues

class CustomSimulationTypeExtractor(SimulationTypeExtractorInterface):
    def extract_simulation_type_values(self, batch, pred) -> SimulationTypeValues:
        # Extract simulation type-specific data
        return SimulationTypeValues(true_type_seqs, sim_type_seqs, sim_mask)
```

### Using Custom Components

```python
# Inject custom extractors for predictions
custom_pred_computer = PredictionMetricsComputer(
    num_event_types=5,
    time_extractor=CustomTimeExtractor(),
    type_extractor=CustomTypeExtractor()
)

# Inject custom extractors for simulations
custom_sim_computer = SimulationMetricsComputer(
    num_event_types=5,
    time_extractor=CustomSimulationTimeExtractor(),
    type_extractor=CustomSimulationTypeExtractor()
)

# Use with MetricsHelper
helper = MetricsHelper(
    num_event_types=5,
    prediction_computer=custom_pred_computer,
    simulation_computer=custom_sim_computer
)
```

## File Structure

```
metrics_helper/
├── __init__.py                           # Package exports with all interfaces
├── interfaces.py                         # Abstract interfaces (general, time, type, simulation)
├── shared_types.py                       # Data containers and enums for all modes
├── main_metrics_helper.py                # Main orchestrator with strategy pattern
├── prediction_metrics_computer.py        # Prediction metrics with time/type separation
├── simulation_metrics_computer.py        # Simulation metrics with time/type separation
└── README.md                            # This file
```

## Key Features

### Enhanced Type Safety
- Separate data containers for time (`TimeValues`) and type (`TypeValues`) metrics
- Type-safe enum-based metric selection
- Clear separation of concerns between different metric categories

### Granular Computation
- Compute all metrics: `compute_all_metrics()`
- **Prediction mode:**
  - Compute only time metrics: `compute_all_time_metrics()`
  - Compute only type metrics: `compute_all_type_metrics()`
- **Simulation mode:**
  - Compute only time metrics: `compute_all_simulation_time_metrics()`
  - Compute only type metrics: `compute_all_simulation_type_metrics()`
- Selective metric computation via configuration

### Flexible Architecture
- Strategy pattern for easy mode switching (prediction/simulation)
- Dependency injection for custom computers and extractors
- Interface segregation for focused, maintainable code

### Performance Optimization
- Selective computation reduces unnecessary calculations
- Specialized extractors minimize data processing overhead
- Cached metric availability checks

This modular design makes the metrics system more maintainable, testable, and extensible while preserving backward compatibility.
