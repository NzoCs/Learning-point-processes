# EasyTPP CLI - Command Line Interface

EasyTPP CLI v4.0 - Temporal Point Processes with runners architecture

## Overview

The EasyTPP CLI provides a command-line interface for running temporal point process experiments, data inspection, synthetic data generation, system information, and benchmarking.

## Installation

Make sure you have EasyTPP installed and activated:

```bash
# Activate your virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# The CLI should be available as 'new-ltpp'
new-ltpp --help
```

## Available Commands

### `new-ltpp run` - Run TPP Experiments

Run a complete temporal point process experiment with training, testing, and prediction phases.

```bash
# Basic usage with defaults
new-ltpp run

# Run with custom configurations
new-ltpp run \
  --config path/to/config.yaml \
  --data-config test \
  --model-config neural_small \
  --training-config quick_test \
  --model NHP \
  --phase all

# Run only training phase
new-ltpp run --phase train --epochs 100

# Run with GPU
new-ltpp run --gpu 0
```

**Options:**

- `--config, -c`: YAML configuration file
- `--data-config`: Data configuration (test, large, synthetic)
- `--model-config`: Model configuration (neural_small, neural_large)
- `--training-config`: Training configuration (quick_test, full_training)
- `--data-loading-config`: Data loading configuration
- `--simulation-config`: Simulation configuration
- `--thinning-config`: Thinning configuration
- `--logger-config`: Logger configuration (tensorboard, mlflow, csv)
- `--model, -m`: Model ID (NHP, RMTPP, etc.)
- `--phase, -p`: Execution phase (train/test/predict/all)
- `--epochs, -e`: Maximum number of epochs
- `--save-dir, -s`: Save directory
- `--gpu, -g`: GPU id to use

### `new-ltpp inspect` - Data Inspection and Visualization

Inspect and visualize temporal point process data with comprehensive analysis.

```bash
# Basic data inspection
new-ltpp inspect ./data/my_dataset

# Advanced inspection with custom options
new-ltpp inspect ./data/my_dataset \
  --format json \
  --output ./analysis_results \
  --save \
  --show \
  --max-seq 1000
```

**Options:**

- `data_dir`: Directory containing the data (required)
- `--format, -f`: Data format (json, csv, etc.)
- `--output, -o`: Output directory for results
- `--save/--no-save`: Save analysis plots
- `--show/--no-show`: Display plots
- `--max-seq`: Maximum number of sequences to analyze

### `new-ltpp generate` - Synthetic Data Generation

Generate synthetic temporal point process data for testing and experimentation.

```bash
# Generate basic synthetic data
new-ltpp generate

# Generate with custom parameters
new-ltpp generate \
  --output ./synthetic_data \
  --num-seq 5000 \
  --max-len 200 \
  --event-types 5 \
  --method nhp \
  --seed 42
```

**Options:**

- `--output, -o`: Output directory
- `--num-seq, -n`: Number of sequences
- `--max-len, -l`: Maximum sequence length
- `--event-types, -t`: Number of event types
- `--method, -m`: Generation method
- `--config, -c`: Configuration file
- `--seed`: Random seed

### `new-ltpp info` - System Information

Display comprehensive system information including hardware, software, and dependencies.

```bash
# Basic system info
new-ltpp info

# Include/exclude specific information
new-ltpp info --deps --hw --output system_report.txt
```

**Options:**

- `--deps/--no-deps`: Include dependencies (default: True)
- `--hw/--no-hw`: Include hardware info (default: True)
- `--output, -o`: Output file for report

### `new-ltpp setup` - Interactive Configuration

Run interactive setup wizard for configuring experiments and environments.

```bash
# Basic interactive setup
new-ltpp setup

# Setup with specific type and options
new-ltpp setup \
  --type experiment \
  --output my_config.yaml \
  --quick
```

**Options:**

- `--type, -t`: Setup type (experiment, data, model)
- `--output, -o`: Output configuration file
- `--quick, -q`: Quick mode with defaults

### `new-ltpp benchmark` - Performance Benchmarking

Run comprehensive benchmarks comparing different models and configurations.

```bash
# List available benchmarks
new-ltpp benchmark --list

# Run all benchmarks on test data
new-ltpp benchmark --config benchmark_config.yaml --data-config test --all

# Run specific benchmarks
new-ltpp benchmark --config config.yaml --benchmarks accuracy latency --data-config test large

# Run benchmarks on all data configurations
new-ltpp benchmark --config config.yaml --all --all-configs
```

**Options:**

- `--config, -c`: Configuration file (required)
- `--data-config`: Data configuration(s) - can be repeated
- `--benchmarks, -b`: List of benchmarks to run
- `--output, -o`: Output directory
- `--run-all`: Run all available benchmarks
- `--run-all-configs`: Run on all data configurations
- `--list`: List available benchmarks

### `new-ltpp version` - Show Version

Display CLI version and architecture information.

```bash
new-ltpp version
```

## Examples

### Complete Experiment Pipeline

```bash
# 1. Inspect your data
new-ltpp inspect ./data/my_experiment --save

# 2. Run a quick test experiment
new-ltpp run --data-config test --training-config quick_test --phase all

# 3. Generate synthetic data for testing
new-ltpp generate --output ./synthetic --num-seq 1000

# 4. Run benchmarks
new-ltpp benchmark --config benchmark.yaml --data-config test --all

# 5. Check system information
new-ltpp info --output system_check.txt
```

### Configuration Files

The CLI supports YAML configuration files for complex setups:

```yaml
# config.yaml
data_config: test
model_config: neural_small
training_config: standard
simulation_config: simulation_fast
logger_config: tensorboard
```

```bash
new-ltpp run --config config.yaml
```

## Help and Documentation

Get help for any command:

```bash
# Main help
new-ltpp --help

# Command-specific help
new-ltpp run --help
new-ltpp inspect --help
new-ltpp benchmark --help
```

## Architecture

EasyTPP CLI v4.0 uses a modular "runners" architecture:

- **ExperimentRunner**: Handles complete experiments
- **DataInspector**: Data analysis and visualization
- **DataGenerator**: Synthetic data generation
- **SystemInfo**: System diagnostics
- **InteractiveSetup**: Guided configuration
- **BenchmarkRunner**: Performance benchmarking

Each runner is independent and can be used programmatically or via CLI.

## Troubleshooting

### Common Issues

1. **Command not found**: Make sure your virtual environment is activated
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **GPU not available**: Use `--gpu -1` for CPU-only execution
4. **Configuration errors**: Validate your YAML files with `easytpp setup`

### Debug Mode

Add `--debug` to any command for verbose output:

```bash
new-ltpp run --debug --phase train
```

## Contributing

When adding new CLI commands:

1. Create a new runner class in `new_ltpp/runners/`
2. Add the command to `new_ltpp/scripts/cli.py`
3. Add corresponding tests in `tests/scripts/test_cli.py`
4. Update this README

## License

See main project license file.
