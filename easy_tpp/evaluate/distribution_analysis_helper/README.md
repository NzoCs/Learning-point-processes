# Distribution Analysis Helper

A comprehensive toolkit for analyzing and comparing temporal point process distributions. This module provides statistical analysis, visualization, and comparison capabilities for evaluating TPP models against ground truth data.

## Overview

This package implements a modular architecture following SOLID principles to ensure maintainable and extensible code. It provides tools for extracting data from various sources, generating visualizations, calculating metrics, and comparing distributions between ground truth and simulated temporal point processes.

## Key Features

- **Multi-source Data Extraction**: Extract data from TPPDataset objects, DataLoaders, or simulation results
- **Statistical Analysis**: Comprehensive distribution analysis with density plots and regression analysis  
- **Visualization**: Generate publication-ready plots for inter-event times, event types, sequence lengths, and cross-correlations
- **Metrics Calculation**: Compute summary statistics and comparison metrics
- **Modular Design**: Extensible architecture following the Open/Closed Principle

## Core Components

### Data Extractors (`data_extractors.py`)
- **`TPPDatasetExtractor`**: High-performance extraction directly from TPPDataset objects
- **`LabelDataExtractor`**: Extracts ground truth data from DataLoaders
- **`SimulationDataExtractor`**: Processes simulation results and predictions

### Plot Generators (`plot_generators.py`)
Specialized plot generators extending `BasePlotGenerator`:
- **`InterEventTimePlotGenerator`**: Distribution plots for time intervals between events
- **`EventTypePlotGenerator`**: Event type frequency analysis and comparison
- **`SequenceLengthPlotGenerator`**: Sequence length distribution analysis
- **`CrossCorrelationPlotGenerator`**: Cross-correlation analysis between datasets

### Analysis Tools
- **`DistributionAnalyzer`** (`distribution_analyzer.py`): Statistical analysis utilities with density plotting and regression analysis
- **`MetricsCalculatorImpl`** (`metrics_calculator.py`): Comprehensive metrics calculation for TPP data
- **`TemporalPointProcessComparator`** (`comparator.py`): Main orchestrator class for complete analysis workflows

### Interfaces (`interfaces.py`)
Protocol-based interfaces ensuring clean separation of concerns:
- **`DataExtractor`**: Interface for data extraction components
- **`PlotGenerator`**: Interface for plot generation components  
- **`MetricsCalculator`**: Interface for metrics calculation components

## Usage Example

```python
from easy_tpp.evaluate.distribution_analysis_helper import (
    TemporalPointProcessComparator,
    TPPDatasetExtractor,
    SimulationDataExtractor,
    MetricsCalculatorImpl
)
from easy_tpp.evaluate.distribution_analysis_helper.plot_generators import (
    InterEventTimePlotGenerator,
    EventTypePlotGenerator,
    SequenceLengthPlotGenerator
)

# Initialize extractors
label_extractor = TPPDatasetExtractor(ground_truth_dataset)
simulation_extractor = SimulationDataExtractor(simulation_results)

# Set up plot generators
plot_generators = [
    InterEventTimePlotGenerator(),
    EventTypePlotGenerator(), 
    SequenceLengthPlotGenerator()
]

# Initialize metrics calculator
metrics_calculator = MetricsCalculatorImpl()

# Create comparator and run analysis
comparator = TemporalPointProcessComparator(
    label_extractor=label_extractor,
    simulation_extractor=simulation_extractor,
    plot_generators=plot_generators,
    metrics_calculator=metrics_calculator,
    output_dir="./analysis_results"
)

# Results automatically saved to output directory
```

## Output

The analysis generates:
- **Visualization plots**: Density comparisons, distribution analyses, correlation plots
- **Metrics JSON**: Comprehensive statistical summaries and comparison metrics
- **Regression analysis**: Statistical relationship analysis between distributions

## Extension

To add new analysis capabilities:

1. **New extractors**: Implement the `DataExtractor` protocol
2. **New plot types**: Extend `BasePlotGenerator` class
3. **New metrics**: Extend `MetricsCalculator` protocol
4. **New analysis**: Add methods to `DistributionAnalyzer`

## Dependencies

- NumPy: Numerical computations
- Matplotlib: Plotting and visualization
- Seaborn: Statistical visualization enhancements  
- SciPy: Statistical analysis and regression
- PyTorch: Tensor operations (for data extraction)

## Visualization Capabilities

The module provides comprehensive visualization tools for temporal point process analysis through specialized plot generators:

### 1. Inter-Event Time Distribution Plots (`InterEventTimePlotGenerator`)

**Purpose**: Analyzes the distribution of time intervals between consecutive events.

**Generated Plots**:
- **Density Histogram**: Log-scale density comparison between ground truth and simulation data
- **Regression Analysis**: Linear regression on log-transformed densities to identify exponential patterns
- **QQ Plot**: Quantile-quantile comparison to assess distributional similarity (log-scale)

**Key Features**:
- Statistical annotations with mean, median, and standard deviation
- Automatic regression line fitting with R² values and slope coefficients
- Color-coded comparison (ground truth in blue, simulation in red)

### 2. Event Type Distribution Plots (`EventTypePlotGenerator`)

**Purpose**: Compares the frequency distribution of different event types between datasets.

**Generated Plots**:
- **Bar Chart Comparison**: Side-by-side probability bars for each event type
- **Statistical Summary**: Top 3 most frequent event types with probabilities

**Key Features**:
- Normalized probability display for fair comparison
- Automatic identification of dominant event types
- Grid layout for easy visual comparison

### 3. Sequence Length Distribution Plots (`SequenceLengthPlotGenerator`)

**Purpose**: Analyzes the distribution of sequence lengths in temporal point processes.

**Generated Plots**:
- **Histogram Comparison**: Density-normalized histograms comparing sequence lengths
- **Statistical Annotations**: Mean, median, and standard deviation for both datasets
- **QQ Plot**: Quantile-quantile comparison for sequence lengths (linear scale)

**Key Features**:
- Automatic bin optimization for optimal visualization
- Statistical summary boxes for quick comparison
- Robust handling of varying sequence length ranges

### 4. Cross-Correlation Plots (`CrossCorrelationPlotGenerator`)

**Purpose**: Analyzes temporal dependencies and correlation patterns in event occurrence.

**Generated Plots**:
- **Cross-Correlation Function**: Shows correlation between counting process increments at different time lags
- **Time Lag Analysis**: Reveals temporal dependencies and clustering patterns

**Key Features**:
- Configurable window sizes for correlation analysis
- Mathematical annotation explaining the correlation measure
- Zero-lag reference line for temporal symmetry assessment

### 5. Advanced Distribution Analysis (`DistributionAnalyzer`)

**Additional Visualization Methods**:

#### Density Comparison Plots
- **Multi-dataset Support**: Compare multiple distributions simultaneously
- **Regression Analysis**: Automatic power-law and exponential fitting
- **Dynamic Thresholding**: Adaptive threshold for regression analysis
- **Log-scale Visualization**: Optimized for heavy-tailed distributions

#### QQ Plots (Quantile-Quantile)
- **Linear and Log-scale Options**: Flexible scaling based on data characteristics
- **Robust Quantile Computation**: Efficient calculation for large datasets
- **Interpretive Annotations**: Built-in explanations for plot interpretation
- **Memory Optimization**: Efficient handling of large datasets

### Plot Customization Features

**Visual Styling**:
- Professional publication-ready formatting
- Consistent color schemes across all plot types
- High-resolution output (300 DPI) for publication quality
- Seaborn integration for enhanced aesthetics

**Statistical Annotations**:
- Automatic calculation and display of summary statistics
- Regression analysis with goodness-of-fit metrics
- Dynamic positioning to avoid overlap
- Transparent background boxes for readability

**Error Handling**:
- Graceful handling of insufficient data
- Automatic fallback for edge cases
- Comprehensive logging for debugging
- Memory-efficient processing for large datasets

### Example Output Files

When running a complete analysis, the following plot files are generated:
```
output_dir/
├── inter_event_time_comparison.png
├── inter_event_time_comparison_qq.png
├── event_type_distribution.png
├── sequence_length_comparison.png
├── sequence_length_comparison_qq.png
└── cross_correlation_analysis.png
```

Each plot is saved with descriptive filenames and includes both main distribution plots and supplementary QQ plots where applicable.
