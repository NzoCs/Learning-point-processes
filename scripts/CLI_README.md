# 🚀 EasyTPP CLI - Comprehensive Temporal Point Process Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CLI Version](https://img.shields.io/badge/CLI%20Version-2.1-brightgreen.svg)](#)

A modern, comprehensive command-line interface for Temporal Point Process (TPP) research with EasyTPP. Now includes data generation, inspection, and benchmarking capabilities.

## ✨ Key Features

- 🎯 **Comprehensive CLI**: Intuitive commands with complete help system
- 🎨 **Rich Terminal Output**: Colored tables, progress bars, and styled display
- 🔧 **Interactive Mode**: Step-by-step guided configuration
- 📊 **Data Generation**: Synthetic TPP data using various simulators
- 🔍 **Data Inspection**: Comprehensive data visualization and analysis
- 🎯 **Benchmarking**: Performance evaluation with multiple benchmark types
- 📁 **Configuration Management**: Template system and validation
- 🚀 **Multiple Execution Modes**: Train, test, predict, validate and more
- 🛡️ **Robust Error Handling**: Detailed error reports and recovery
- 📊 **Advanced Logging**: Rotation and log retention
- 🔄 **Multi-Platform**: Compatible with Windows, macOS and Linux

## 📦 Quick Installation

1. **Install dependencies**:

   ```bash
   python setup_cli.py
   ```

2. **Verify installation**:

   ```bash
   python easytpp_cli.py --version
   ```

3. **Quick test**:

   ```bash
   python easytpp_cli.py --help
   ```

## 🎯 Quick Start

### Interactive Mode (Recommended for beginners)

```bash
python easytpp_cli.py interactive
```

Interactive mode guides you through:

- Configuration file selection
- Experiment and dataset selection
- Parameter configuration
- Execution confirmation

### Basic Usage

```bash
# Run a test experiment
python easytpp_cli.py run --config configs/runner_config.yaml --experiment THP --dataset H2expc --phase test

# Generate synthetic data
python easytpp_cli.py data-gen --type hawkes --num-sims 100 --output ./my_data

# Inspect and visualize data
python easytpp_cli.py data-inspect --config ./config.yaml --experiment H2expi

# Run benchmarks
python easytpp_cli.py benchmark --type mean --dataset test

# List available configurations
python easytpp_cli.py list-configs --dir configs

# Validate a configuration
python easytpp_cli.py validate --config configs/runner_config.yaml --experiment THP --dataset H2expc
```

### Advanced Usage

```bash
# Training with checkpoint and custom output directory
python easytpp_cli.py run \
  --config configs/advanced_config.yaml \
  --experiment TransformerHP \
  --dataset taxi \
  --phase train \
  --checkpoint checkpoints/model_epoch_50.ckpt \
  --output experiments/run_001 \
  --device gpu \
  --seed 42

# Run all phases sequentially
python easytpp_cli.py run \
  --config configs/full_pipeline.yaml \
  --experiment THP \
  --dataset H2expc \
  --phase all

# Generate Hawkes process data with custom parameters
python easytpp_cli.py data-gen \
  --type hawkes \
  --num-sims 500 \
  --dim 3 \
  --start 0 \
  --end 200 \
  --output ./experiments/synthetic_data

# Run comprehensive benchmarks
python easytpp_cli.py benchmark \
  --all-benchmarks \
  --all-datasets \
  --output ./benchmark_results

# Data inspection with custom output
python easytpp_cli.py data-inspect \
  --config ./configs/inspection_config.yaml \
  --experiment H2expi \
  --split train \
  --output ./visualizations
```

## 📋 Available Commands

### Core Commands

#### `run` - Execute TPP Experiments

Execute temporal point process experiments with comprehensive configuration options.

```bash
python easytpp_cli.py run [OPTIONS]
```

**Options**:

- `--config, -c`: Path to YAML configuration file (required)
- `--experiment, -e`: Experiment ID in config file (required)
- `--dataset, -d`: Dataset ID in config file (required)
- `--phase, -p`: Phase to execute [train|test|predict|validation|all] (default: test)
- `--checkpoint`: Path to checkpoint file
- `--output`: Output directory for results
- `--device`: Device to use [cpu|gpu|auto] (default: auto)
- `--seed`: Random seed for reproducibility

#### `interactive` - Interactive Configuration Mode

Launch interactive mode for guided experiment setup.

```bash
python easytpp_cli.py interactive
```

### Data Generation Commands

#### `data-gen` (alias: `gen`) - Generate Synthetic TPP Data

Generate synthetic temporal point process data using various simulators.

```bash
python easytpp_cli.py data-gen [OPTIONS]
python easytpp_cli.py gen [OPTIONS]  # Short alias
```

**Options**:

- `--type, -t`: Generator type [hawkes|selfcorrecting] (default: hawkes)
- `--output, -o`: Output directory (default: ./data/generated)
- `--num-sims, -n`: Number of simulations (default: 10)
- `--start`: Start time (default: 0.0)
- `--end`: End time (default: 100.0)
- `--dim`: Process dimension (default: 2)
- `--config, -c`: Custom config file
- `--verbose, -v`: Verbose output

**Examples**:

```bash
# Generate 100 Hawkes process sequences
python easytpp_cli.py data-gen --type hawkes --num-sims 100 --dim 3

# Generate self-correcting process data
python easytpp_cli.py gen --type selfcorrecting --num-sims 50 --output ./sc_data
```

### Data Analysis Commands

#### `data-inspect` (alias: `inspect`) - Inspect and Visualize TPP Data

Comprehensive data visualization and analysis for TPP datasets.

```bash
python easytpp_cli.py data-inspect [OPTIONS]
python easytpp_cli.py inspect [OPTIONS]  # Short alias
```

**Options**:

- `--config, -c`: Configuration file path (default: ./main/data_inspection/config.yaml)
- `--experiment, -e`: Experiment ID (default: H2expi)
- `--output, -o`: Output directory for visualizations (default: ./visu)
- `--split`: Data split to visualize [train|test|dev] (default: test)
- `--verbose, -v`: Verbose output

**Examples**:

```bash
# Inspect test data with default config
python easytpp_cli.py data-inspect

# Inspect training data with custom output
python easytpp_cli.py inspect --split train --output ./my_visualizations

# Inspect specific experiment
python easytpp_cli.py data-inspect --experiment MyExperiment --config ./my_config.yaml
```

### Benchmarking Commands

#### `benchmark` (alias: `bench`) - Run Performance Benchmarks

Execute comprehensive performance benchmarks on TPP datasets.

```bash
python easytpp_cli.py benchmark [OPTIONS]
python easytpp_cli.py bench [OPTIONS]  # Short alias
```

**Options**:

- `--config, -c`: Benchmark configuration file (default: ./main/run_benchmarks/bench_config.yaml)
- `--dataset, -d`: Dataset name
- `--type, -t`: Benchmark type [mean|mark_distribution|intertime_distribution|last_mark]
- `--all-datasets`: Run on all datasets
- `--all-benchmarks`: Run all benchmarks
- `--output, -o`: Output directory (default: ./benchmark_results)
- `--list-datasets`: List available datasets
- `--list-benchmarks`: List available benchmarks
- `--verbose, -v`: Verbose output

**Available Benchmarks**:

- `mean`: Mean Inter-Time Benchmark - predicts mean inter-arrival time
- `mark_distribution`: Mark Distribution Benchmark - samples marks from training distribution
- `intertime_distribution`: Inter-Time Distribution Benchmark - samples inter-times from training distribution
- `last_mark`: Last Mark Benchmark - predicts the last observed mark

**Examples**:

```bash
# List available datasets and benchmarks
python easytpp_cli.py benchmark --list-datasets
python easytpp_cli.py bench --list-benchmarks

# Run specific benchmark on specific dataset
python easytpp_cli.py benchmark --type mean --dataset taxi

# Run all benchmarks on specific dataset
python easytpp_cli.py bench --all-benchmarks --dataset test

# Run specific benchmark on all datasets
python easytpp_cli.py benchmark --type mean --all-datasets

# Run comprehensive benchmarking
python easytpp_cli.py bench --all-benchmarks --all-datasets --output ./full_benchmark_results
```

### Configuration Commands

#### `list-configs` - List Available Configurations

Display all available configuration files in a directory.

```bash
python easytpp_cli.py list-configs [--dir DIRECTORY]
```

#### `validate` - Validate Configuration

Validate a configuration file and display summary.

```bash
python easytpp_cli.py validate --config CONFIG --experiment EXP --dataset DATA
```

#### `info` - System Information

Display system and environment information including PyTorch, CUDA and hardware.

```bash
python easytpp_cli.py info
```

## ⚙️ Configuration

### Configuration File Structure

YAML configuration files follow this structure:

```yaml
base_config:
  seed: 42
  device: auto
  log_level: INFO

experiments:
  THP:
    model_config:
      model_name: THP
      hidden_size: 64
      num_layers: 4
      # ... more model parameters
      
    data_config:
      dataset_name: H2expc
      batch_size: 32
      # ... more data parameters
      
    training_config:
      optimizer: Adam
      learning_rate: 0.001
      # ... more training parameters

datasets:
  H2expc:
    path: ./data/h2expc
    type: point_process
    # ... dataset configuration
```

### Data Generation Configuration

For synthetic data generation, you can use custom parameter files:

```yaml
hawkes_config:
  mu: [0.2, 0.2]
  alpha: [[0.4, 0.0], [0.0, 0.8]]
  beta: [[1.0, 0.0], [0.0, 2.0]]
  dim_process: 2
  time_range: [0, 100]
  
selfcorrecting_config:
  dim_process: 1
  time_range: [0, 200]
  # ... other parameters
```

### Benchmark Configuration

Benchmark configurations are defined in `bench_config.yaml`:

```yaml
data:
  test:
    data_format: pkl
    data_specs:
      num_event_types: 5
      pad_token_id: 0
    dataset_id: test
    
  taxi:
    data_format: pkl
    data_specs:
      num_event_types: 2
      pad_token_id: 0
    dataset_id: taxi
    # ... more datasets

benchmark_settings:
  output_format: json
  save_predictions: true
  compute_metrics: true
```

### CLI Configuration

Global CLI parameters can be configured in `configs/cli_config.yaml`:

```yaml
cli:
  default_config_dir: ./configs
  default_output_dir: ./outputs
  use_rich: true
  color_theme: default

defaults:
  experiment_id: THP
  dataset_id: H2expc
  phase: test
```

## 🎨 Rich Terminal Features

When the `rich` package is installed, the CLI provides:

- **Colored Output**: Syntax highlighting and colored text
- **Progress Bars**: Real-time progress tracking
- **Tables**: Formatted data display
- **Panels**: Organized information display
- **Interactive Prompts**: Enhanced user input

## 🛠️ Helper Scripts

### Windows

- **PowerShell**: `easytpp.ps1` - PowerShell script
- **Batch**: `easytpp.bat` - Simple batch script
- **Modern**: `easytpp_modern_cli.py` - Alternative CLI with Typer

### Unix/Linux/macOS

- **Shell**: `easytpp` - Executable shell script
- **Python**: `easytpp_cli.py` - Main script

## 📁 Directory Structure

```text
New_LTPP/
├── easytpp_cli.py              # Main CLI
├── easytpp_modern_cli.py       # Alternative CLI (Typer)
├── setup_cli.py               # Installation script
├── easytpp.ps1                # PowerShell wrapper
├── easytpp.bat                # Batch wrapper
├── requirements-cli.txt        # CLI dependencies
├── CLI_README.md              # Detailed documentation
├── configs/
│   ├── runner_config_template.yaml
│   ├── cli_config.yaml
│   └── bench_config.yaml
├── examples/
│   ├── basic_example.py
│   └── advanced_pipeline.py
├── main/
│   ├── data_gen/              # Data generation scripts
│   ├── data_inspection/       # Data analysis scripts
│   └── run_benchmarks/        # Benchmark scripts
├── outputs/                   # Experiment results
├── logs/                      # Log files
├── benchmark_results/         # Benchmark outputs
├── data/
│   └── generated/             # Generated datasets
├── visu/                      # Visualizations
└── checkpoints/               # Saved models
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-cli.txt
   ```

2. **Configuration Not Found**: Check the config file path

   ```bash
   python easytpp_cli.py list-configs
   ```

3. **CUDA Issues**: Check device availability

   ```bash
   python easytpp_cli.py info
   ```

4. **Data Generation Issues**: Ensure output directory permissions

   ```bash
   # Check directory permissions
   python easytpp_cli.py data-gen --output ./test_data --num-sims 1
   ```

5. **Benchmark Configuration**: Verify benchmark config file

   ```bash
   python easytpp_cli.py benchmark --list-datasets
   python easytpp_cli.py benchmark --list-benchmarks
   ```

### Verbose Mode

Enable verbose logging for debugging:

```bash
python easytpp_cli.py --verbose run --config config.yaml --experiment THP --dataset H2expc
python easytpp_cli.py data-gen --verbose --type hawkes --num-sims 10
python easytpp_cli.py benchmark --verbose --type mean --dataset test
```

## 📊 Usage Examples

### Example 1: Complete TPP Workflow

```bash
# Step 1: Generate synthetic data
python easytpp_cli.py data-gen \
  --type hawkes \
  --num-sims 1000 \
  --dim 2 \
  --output ./experiments/synthetic_hawkes

# Step 2: Inspect the generated data
python easytpp_cli.py data-inspect \
  --config ./configs/inspection_config.yaml \
  --experiment synthetic_data \
  --output ./visualizations/hawkes

# Step 3: Train a model on the data
python easytpp_cli.py run \
  --config ./configs/training_config.yaml \
  --experiment THP \
  --dataset synthetic_hawkes \
  --phase train \
  --output ./experiments/thp_synthetic

# Step 4: Evaluate the model
python easytpp_cli.py run \
  --config ./configs/training_config.yaml \
  --experiment THP \
  --dataset synthetic_hawkes \
  --phase test \
  --checkpoint ./experiments/thp_synthetic/checkpoints/best.ckpt

# Step 5: Run benchmarks for comparison
python easytpp_cli.py benchmark \
  --all-benchmarks \
  --dataset synthetic_hawkes \
  --output ./benchmark_results/synthetic_comparison
```

### Example 2: Quick Model Training

```bash
# Start interactive mode for guided setup
python easytpp_cli.py interactive
```

### Example 3: Automated Pipeline

```bash
# Train a model
python easytpp_cli.py run -c configs/thp_config.yaml -e THP -d taxi -p train --output experiments/taxi_run_001

# Evaluate the trained model
python easytpp_cli.py run -c configs/thp_config.yaml -e THP -d taxi -p test --checkpoint experiments/taxi_run_001/checkpoints/best.ckpt

# Compare with benchmarks
python easytpp_cli.py benchmark --all-benchmarks --dataset taxi --output experiments/taxi_benchmarks
```

### Example 4: Data Analysis Workflow

```bash
# Generate multiple types of synthetic data
python easytpp_cli.py gen --type hawkes --num-sims 500 --output ./data/hawkes_large
python easytpp_cli.py gen --type selfcorrecting --num-sims 500 --output ./data/sc_large

# Inspect both datasets
python easytpp_cli.py inspect --experiment hawkes_analysis --output ./analysis/hawkes
python easytpp_cli.py inspect --experiment sc_analysis --output ./analysis/selfcorrecting

# Run comprehensive benchmarks
python easytpp_cli.py bench --all-benchmarks --all-datasets --output ./analysis/full_benchmark
```

### Example 5: Configuration Management

```bash
# List all available configurations
python easytpp_cli.py list-configs

# Validate a specific configuration
python easytpp_cli.py validate -c configs/new_config.yaml -e MyExperiment -d MyDataset

# Run with validation
python easytpp_cli.py run -c configs/new_config.yaml -e MyExperiment -d MyDataset -p train
```

## 🚀 Advanced Features

### Modern CLI with Typer

Use the modern version of the CLI for an enhanced experience:

```bash
python easytpp_modern_cli.py run --help
python easytpp_modern_cli.py interactive
python easytpp_modern_cli.py data-gen --type hawkes
```

### Command Aliases

Use shorter command aliases for faster access:

```bash
# These are equivalent
python easytpp_cli.py data-gen --type hawkes
python easytpp_cli.py gen --type hawkes

# These are equivalent  
python easytpp_cli.py data-inspect --experiment H2expi
python easytpp_cli.py inspect --experiment H2expi

# These are equivalent
python easytpp_cli.py benchmark --type mean
python easytpp_cli.py bench --type mean
```

### Batch Processing

Process multiple configurations or datasets:

```bash
# Process multiple datasets with the same benchmark
for dataset in taxi H2expc test; do
  python easytpp_cli.py benchmark --type mean --dataset $dataset --output results_$dataset
done

# Generate multiple synthetic datasets
for i in {1..5}; do
  python easytpp_cli.py gen --num-sims 100 --output ./data/batch_$i --seed $i
done
```

### Example Scripts

Explore example scripts in the `examples/` folder:

```bash
cd examples
python basic_example.py
python advanced_pipeline.py
python data_generation_example.py
python benchmark_comparison.py
```

### Custom Configuration

Create your own configuration templates:

1. Copy `configs/runner_config_template.yaml`
2. Modify according to your needs
3. Use with the CLI

```bash
cp configs/runner_config_template.yaml configs/my_custom_config.yaml
# Edit the file...
python easytpp_cli.py run --config configs/my_custom_config.yaml --experiment MyExp --dataset MyData
```

## 📝 Logging

The CLI provides comprehensive logging:

- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Rotation**: Daily rotating logs
- **Retention**: Configurable retention
- **Format**: Structured messages with timestamps

Log files are stored in:
- `./logs/easytpp_cli.log` - Main CLI log
- `./logs/experiments/` - Experiment-specific logs
- `./logs/benchmarks/` - Benchmark logs
- `./logs/data_generation/` - Data generation logs

## 🤝 Contributing

To contribute to the CLI tool:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install CLI in development mode
pip install -e .

# Run tests
python -m pytest tests/cli/

# Check code quality
flake8 easytpp_cli.py
black easytpp_cli.py
```

## 📞 Support

For support and questions:

- Consult the documentation
- Open a GitHub issue
- Contact the EasyTPP team

## 📜 License

This project is licensed under Apache 2.0. See the [LICENSE](LICENSE) file for details.

---

**EasyTPP CLI v2.1** - Making Temporal Point Process research accessible and comprehensive.

### 🎯 Next Steps

1. **Test the CLI**: `python easytpp_cli.py --help`
2. **Interactive Mode**: `python easytpp_cli.py interactive`
3. **Generate Data**: `python easytpp_cli.py data-gen --help`
4. **Inspect Data**: `python easytpp_cli.py data-inspect --help`
5. **Run Benchmarks**: `python easytpp_cli.py benchmark --help`
6. **Read Documentation**: `CLI_README.md`
7. **Try Examples**: `cd examples && python basic_example.py`

### 🚀 Quick Start Commands

```bash
# Complete workflow example
python easytpp_cli.py gen --type hawkes --num-sims 100 --output ./my_data
python easytpp_cli.py inspect --experiment my_analysis --output ./my_visualizations  
python easytpp_cli.py run --config configs/runner_config_template.yaml --experiment THP --dataset H2expc --phase test
python easytpp_cli.py bench --type mean --dataset test --output ./my_benchmarks
```
