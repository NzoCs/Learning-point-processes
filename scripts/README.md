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

- `--config, -c`: YAML configuration file [default: yaml_configs/configs.yaml]
- `--dataset-id`: Data configuration ID (e.g. taxi, taobao, amazon, retweet, test, hawkes1, etc.) [default: test]
- `--general-specs-config`: General model specs configuration (quick_test, debug, h16, h32, etc.) [default: quick_test]
- `--model-specs-config`: Model-specific specs configuration (optional, depends on model)
- `--training-config`: Training configuration (quick_test, debug, e500_b1, etc.) [default: quick_test]
- `--data-loading-config`: Data loading configuration (quick_test, b32_w1, b64_w2, etc.) [default: quick_test]
- `--simulation-config`: Simulation configuration (quick_test, debug, tw30_b5000_b16, etc.) [default: quick_test]
- `--thinning-config`: Thinning configuration (quick_test, debug, e50_s15, etc.) [default: quick_test]
- `--statistical-test-config`: Statistical test configuration [default: quick_test]
- `--logger-config`: Logger configuration (tensorboard, csv, wandb) [default: tensorboard]
- `--model, -m`: Model ID (NHP, RMTPP, etc.) [default: NHP]
- `--phase, -p`: Execution phase (train/test/predict/all) [default: all]
- `--epochs, -e`: Maximum number of epochs overrides [default: 100]
- `--save-dir, -s`: Save directory [default: artifacts]
- `--debug`: Enable verbose debug mode

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

- `data_dir`: Directory containing the data to inspect (required)
- `--format, -f`: Data format (json, csv, hf if saved on Hugging Face) [default: json]
- `--output, -o`: Output directory for analysis results
- `--save / --no-save`: Save analysis plots [default: True]
- `--show / --no-show`: Display plots interactively [default: False]
- `--max-seq`: Maximum number of sequences to analyze
- `--num-event-types`: Explicitly specify the number of event types
- `--debug`: Enable verbose debug mode

### `new-ltpp generate` - Synthetic Data Generation

Generate synthetic temporal point process data for testing and experimentation.

```bash
# Generate basic synthetic data
new-ltpp generate

# Generate with custom parameters
new-ltpp generate \
  --output ./synthetic_data \
  --num-sim 5000 \
  --num-events-per-seq 200 \
  --dim 5 \
  --model hawkes \
  --seed 42
```

**Options:**

- `--output, -o`: Output directory [default: artifacts/generated_data/TIMESTAMP]
- `--num-sim, -n`: Number of sequences to generate [default: 1000]
- `--model, -m`: Generation method (hawkes, self_correcting) [default: hawkes]
- `--dim, -d`: Number of event types/dimensions [default: 2]
- `--burn-in`: Number of events to discard for warmup to reach stationary state [default: 100]
- `--num-events-per-seq`: Number of events to keep per sequence [default: 100]
- `--train-ratio`: Train split ratio [default: 0.6]
- `--test-ratio`: Test split ratio [default: 0.2]
- `--dev-ratio`: Dev split ratio [default: 0.2]
- `--push`: Push the generated dataset directly to Hugging Face Hub
- `--repo-id`: Hugging Face repo ID (e.g., `username/my-dataset`). Required if `--push` is used
- `--private`: Make the Hugging Face dataset private
- `--seed`: Random seed for reproducible generation
- `--mu`: Baseline intensity (JSON list string, e.g. `"[0.2, 0.2]"`)
- `--alpha`: Excitation matrix (JSON list of lists, e.g. `"[[0.4, 0], [0, 0.8]]"`)
- `--beta`: Decay matrix (JSON list of lists)
- `--save-local / --no-local`: Whether to save the dataset locally [default: True]
- `--debug`: Enable verbose debug mode

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

# 3. Generate synthetic data for testing and push directly to Hugging Face
new-ltpp generate --output ./synthetic --num-sim 1000 --num-events-per-seq 100 --burn-in 100 --push --repo-id username/my-dataset

# 4. Run benchmarks
new-ltpp benchmark --config benchmark.yaml --dataset-id test --all

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
