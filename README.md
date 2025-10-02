# New-LTPP: Advanced Temporal Point Process Framework

<div align="center">
  <a href="PyVersion">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.8+-blue.svg">
  </a>
  <a href="LICENSE-CODE">
    <img alt="Code License" src="https://img.shields.io/badge/license-Apache-000000.svg?&color=f5de53">
  </a>
  <a href="commit">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/NzoCs/Learning-point-processes">
  </a>
</div>

<div align="center">
<a href="https://pytorch.org/"> 
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
</a>
<a href="https://lightning.ai/"> 
  <img alt="Lightning" src="https://img.shields.io/badge/Lightning-2.0+-792ee5?logo=pytorch-lightning&logoColor=white" />
</a>
<a href="https://github.com/NzoCs/Learning-point-processes/issues">
  <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/NzoCs/Learning-point-processes" />
</a>
<a href="https://github.com/NzoCs/Learning-point-processes/stargazers">
  <img alt="Stars" src="https://img.shields.io/github/stars/NzoCs/Learning-point-processes" />
</a>
</div>

# EasyTPP - Quick Setup Guide

This guide helps you quickly set up the EasyTPP project using `uv` - the fast Python package manager.

## Prerequisites

- **uv** (replaces pip and venv) - [Install from here](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.8 or higher (uv can install this automatically)
- Git

## Quick Installation with uv

### 1. Clone the project

```bash
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
```

### 2. Setup project with uv (automatically installs in editable mode)

```bash
# Initialize and install dependencies in one command
# This automatically installs the project in editable mode
uv sync

# Or if you want to specify Python version
uv python install 3.11  # Install specific Python version
uv sync --python 3.11   # Use specific Python version
```

### 3. Verify installation

```bash
# Run with uv (automatically uses project environment)
uv run python check_installation.py
```

## Installation Options with uv

Choose the dependency groups that match your needs:

```bash
# Basic installation (main dependencies only)
uv sync

# Include development tools (tests, linting, formatting)
uv sync --group dev

# Include CLI tools (command line interfaces)
uv sync --group cli

# Include documentation tools
uv sync --group docs

# Install all optional dependencies
uv sync --all-extras
```

Note: With `uv sync`, the project is automatically installed in editable mode, so any changes to the source code are immediately reflected.

## Working with uv

### Running commands

```bash
# Run Python scripts (automatically uses project environment)
uv run python your_script.py

# Run pytest
uv run pytest

# Run any command in the project environment
uv run black .
uv run isort .
uv run flake8
```

### Adding new dependencies

```bash
# Add a new dependency
uv add numpy pandas

# Add development dependencies
uv add --dev pytest black isort

# Add optional dependencies to a group
uv add --optional cli rich typer
```

### Environment management

```bash
# Activate shell with project environment
uv shell

# Show project info
uv show

# Update dependencies
uv sync --upgrade
```

## Development Tools Configuration

The project includes preconfigured tools for development, now optimized for `uv`:

### Using uv for development tasks

```bash
# Install development dependencies
uv sync --group dev

# Run development tools with uv
uv run black .          # Format code
uv run isort .          # Organize imports
uv run flake8           # Check code quality
uv run mypy easy_tpp    # Static type checking
uv run pytest          # Run tests with coverage
```

### Makefile (Unix/Linux/Windows with Git Bash)

The project still supports Makefile commands, but they now use `uv`:

**Available Makefile commands:**

```bash
make help          # Show help
make install-all   # Full installation with uv
make check         # Installation verification
make test          # Run tests with uv
make format        # Format code with uv
make lint          # Check code with uv
make clean         # Clean temporary files
make demo          # Quick demonstration
```

### Pre-commit hooks

```bash
# Install pre-commit hooks (uv will handle the environment)
uv run pre-commit install
```

### Available tools

- **black**: Automatic code formatting
- **isort**: Import organization
- **flake8**: Code quality checking
- **mypy**: Static type checking
- **pytest**: Testing with coverage
- **pre-commit**: Git hooks for code quality

### Development workflow with uv

```bash
# One-time setup
uv sync --group dev

# Daily development workflow
uv run black .          # Format your code
uv run isort .          # Sort imports
uv run flake8           # Check code style
uv run mypy easy_tpp    # Check types
uv run pytest          # Run tests

# Add new dependencies as you work
uv add requests         # Add runtime dependency
uv add --dev pytest-mock # Add development dependency
```

## Project Structure

```bash
EasyTemporalPointProcess/
├── pyproject.toml          # Main project configuration
├── uv.lock                 # Lockfile for reproducible builds
├── check_installation.py   # Verification script
├── README.md              # Main documentation
├── easy_tpp/              # Main source code
├── examples/              # Usage examples
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## uv and pyproject.toml Configuration

All project configuration is centralized in `pyproject.toml`, and `uv` uses this for:

- Build system configuration
- Dependencies and optional dependency groups
- Tool configuration (black, isort, pytest, mypy, etc.)
- Project metadata and URLs

`uv` creates a `uv.lock` file for reproducible installations across different environments.

## Dependency Groups

- **`cli`**: Rich terminal interfaces, command line tools
- **`docs`**: Sphinx documentation system, themes and extensions
- **`dev`**: Development workflow tools (tests, linting, formatting)
- **`all`**: Installs all optional dependencies

## Why uv?

`uv` is significantly faster than pip and provides:

- **Fast installations**: 10-100x faster than pip
- **Better dependency resolution**: More reliable than pip
- **Built-in virtual environment management**: No need for separate venv
- **Lock files**: Reproducible builds with `uv.lock`
- **Project management**: Commands like `uv add`, `uv remove`, `uv sync`
- **Python version management**: Can install and switch Python versions
- **Automatic editable installs**: Projects are installed in editable mode by default

## Troubleshooting

### uv not found

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

### Python version error

```bash
# Let uv install Python for you
uv python install 3.11

# Use specific Python version
uv sync --python 3.11
```

### Dependency conflicts

```bash
# Reset and reinstall everything
rm -rf .venv uv.lock  # Unix/macOS
rmdir /s .venv; del uv.lock  # Windows
uv sync
```

### Migration from pip/venv

```bash
# Remove old virtual environment
rm -rf .venv  # Unix/macOS
rmdir /s .venv  # Windows

# Initialize with uv
uv sync

# If you have requirements.txt
uv add -r requirements.txt
```

## Support

If you encounter problems:

1. Make sure `uv` is installed and up to date: `uv --version`
2. Run `uv run python check_installation.py` to diagnose
3. Try `uv sync --reinstall` for a fresh installation
4. Consult the complete documentation in README.md
5. Check [uv documentation](https://docs.astral.sh/uv/) for uv-specific issues
6. Create an issue on GitHub if the problem persists

---

**New-LTPP** is a modern, advanced framework for [Temporal Point Process](https://mathworld.wolfram.com/TemporalPointProcess.html) (TPP) research and development. Originally inspired by [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess), this project has evolved into a comprehensive toolkit with significant enhancements in performance, usability, and research capabilities.

## 🚀 Key Innovations

This framework goes beyond traditional TPP implementations by introducing:

- **🔥 PyTorch Lightning Integration**: Enhanced performance, scalability, and training efficiency
- **📊 Advanced Loss Functions**: Implementation of cutting-edge losses including MMD, Sinkhorn, and Wasserstein distances
- **🎯 Robust Evaluation Metrics**: Comprehensive validation using Wasserstein distance and other advanced metrics
- **🔬 Simulation & Analysis**: Built-in capabilities for model simulation and temporal distribution analysis
- **⚡ Modern Architecture**: Streamlined, modular design with improved maintainability
- **🛠️ Enhanced CLI**: Professional command-line interface with interactive modes
<span id='top'/>

| [Features](#features) | [Project Setup](#project-setup) | [Model List](#model-list) | [Dataset](#dataset) | [Quick Start](#quick-start) | [Benchmark](#benchmark) | [Contributing](#contributing) | [Documentation](#doc) | [Todo List](#todo) | [Citation](#citation) | [Acknowledgment](#acknowledgment) |

## 🆕 What's New
<span id='news'/>

- **[2025-07]** 🔥 **Major Framework Overhaul**: Complete rewrite with PyTorch Lightning integration
- **[2025-07]** 📊 **Advanced Loss Functions**: Added MMD, Sinkhorn, and Wasserstein distance implementations  
- **[2025-07]** 🎯 **Enhanced Evaluation**: Comprehensive validation metrics including distributional analysis
- **[2025-07]** 🛠️ **Modern CLI**: Professional command-line interface with interactive modes
- **[2025-07]** ⚡ **Performance Boost**: Significant training speed improvements through Lightning optimization
- **[2025-07]** 🔬 **Simulation Capabilities**: Built-in model simulation and temporal pattern analysis


## Features
<span id='features'/>

### 🔥 Core Enhancements

- **Lightning-Powered Performance**: Built on PyTorch Lightning for optimized training, automatic mixed precision, and distributed computing support
- **Advanced Loss Functions**: Implementation of cutting-edge losses for better model training:
  - **MMD (Maximum Mean Discrepancy)**: For distribution matching and domain adaptation
  - **Sinkhorn Loss**: Optimal transport-based loss for sequence alignment
  - **Wasserstein Distance**: Earth mover's distance for robust evaluation
- **Comprehensive Evaluation**: Beyond traditional metrics with distributional analysis and temporal pattern validation
- **Modern Architecture**: Clean, modular codebase with enhanced maintainability and extensibility

### ⚡ Performance & Usability

- **Faster Training**: Significant speed improvements through Lightning optimizations
- **Better Resource Management**: Automatic GPU utilization and memory optimization
- **Enhanced CLI**: Professional command-line interface with interactive modes
- **Real-time Monitoring**: Integrated logging and visualization capabilities
- **Flexible Configuration**: YAML-based configuration system with validation

### 🔬 Research Capabilities

- **Model Simulation**: Built-in simulation tools for temporal pattern generation
- **Distribution Analysis**: Comprehensive analysis of generated temporal sequences
- **Robust Evaluation**: Advanced metrics encouraging more thorough model assessment
- **Extensible Framework**: Easy integration of new models, losses, and evaluation metrics


## Project Setup
<span id='project-setup'/>

This project uses modern Python packaging with `pyproject.toml` and `uv` for fast, reliable dependency management. The CLI interface is located in the `scripts/` directory.

### Prerequisites

- **uv** (replaces pip and venv) - [Install from here](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.8 or higher (uv can install this automatically)
- Git

### Quick Setup

```bash
# Clone this repository
git clone https://github.com/NzoCs/Learning-point-processes.git
cd Learning-point-processes

# Setup project with uv (creates virtual environment and installs dependencies)
uv sync

# Install with all optional dependencies
uv sync --all-extras
```

### Installation Options

Choose the dependency groups that fit your needs:

```bash
# Basic installation (core dependencies only)
uv sync

# Include development tools (testing, linting, formatting)
uv sync --group dev

# Include CLI tools (command-line interface)
uv sync --group cli

# Include documentation tools
uv sync --group docs

# Install all optional dependencies
uv sync --all-extras
```

Note: With `uv sync`, the project is automatically installed in editable mode, so changes to the source code are immediately reflected.

### CLI Interface

The project includes a comprehensive CLI interface located in the `scripts/` directory. After installation, you can access the CLI:

```bash
# Navigate to the scripts directory
cd scripts

# Run the main CLI (uv automatically uses the project environment)
uv run python easytpp_cli.py --help

# Quick installation verification
uv run python easytpp_cli.py info

# Interactive mode for guided setup
uv run python easytpp_cli.py interactive
```

### Development Tools

The project includes pre-configured development tools via `pyproject.toml`:

- **Code formatting**: `black` for consistent code style
- **Import sorting**: `isort` for organized imports
- **Linting**: `flake8` for code quality checks
- **Type checking**: `mypy` for static type analysis
- **Testing**: `pytest` with coverage reporting
- **Pre-commit hooks**: `pre-commit` for automated checks

To set up pre-commit hooks:

```bash
# Install pre-commit hooks (after installing dev dependencies)
pre-commit install
```

### Configuration

All project configuration is centralized in `pyproject.toml`:

- Build system configuration
- Dependencies and optional dependency groups
- Tool configurations (black, isort, pytest, mypy, etc.)
- Project metadata and URLs

### Dependency Groups Explained

- **`cli`**: Rich terminal interfaces, command-line tools, progress bars
- **`docs`**: Sphinx documentation system, themes, and extensions
- **`dev`**: Development workflow tools (testing, linting, formatting)
- **`all`**: Installs all optional dependencies for complete functionality

### Verification

After installation, verify everything is working:

```bash
# Run the installation check script
uv run python check_installation.py

# Test the CLI interface
cd scripts
uv run python easytpp_cli.py --version
uv run python easytpp_cli.py info
```


## Model List
<span id='model-list'/>

New-LTPP implements state-of-the-art TPP models with modern PyTorch implementations and enhanced evaluation capabilities:

| No  | Publication |     Model     | Paper                                                                                                                                    | Implementation                                                                                                   |
|:---:|:-----------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
|  1  |   KDD'16    |     RMTPP     | [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) | [Model](easy_tpp/models/rmtpp.py)                   |
|  2  | NeurIPS'17  |      NHP      | [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)                     | [Model](easy_tpp/models/nhp.py)                       |
|  3  | NeurIPS'19  |    FullyNN    | [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690)                                | [Model](easy_tpp/models/fullynn.py)                |
|  4  |   ICML'20   |     SAHP      | [Self-Attentive Hawkes process](https://arxiv.org/abs/1907.07561)                                                                        | [Model](easy_tpp/models/sahp.py)                     |
|  5  |   ICML'20   |      THP      | [Transformer Hawkes process](https://arxiv.org/abs/2002.09291)                                                                           | [Model](easy_tpp/models/thp.py)                       |
|  6  |   ICLR'20   | IntensityFree | [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127)                                                  | [Model](easy_tpp/models/intensity_free.py) |
|  7  |   ICLR'21   |    ODETPP     | [Neural Spatio-Temporal Point Processes (simplified)](https://arxiv.org/abs/2011.04583)                                                  | [Model](easy_tpp/models/ode_tpp.py)               |
|  8  |   ICLR'22   |    AttNHP     | [Transformer Embeddings of Irregularly Spaced Events and Their Participants](https://arxiv.org/abs/2201.00044)                           | [Model](easy_tpp/models/attnhp.py)                 |
|  9  |   Custom    |    Hawkes     | Classical Hawkes Process implementation                                                                                                     | [Model](easy_tpp/models/hawkes.py)                |
| 10  |   Custom    | SelfCorrect   | Self-Correcting Point Process                                                                                                               | [Model](easy_tpp/models/self_correcting.py)       |

### 🆕 Enhanced Loss Functions

- **MMD Loss**: Maximum Mean Discrepancy for distribution matching
- **Sinkhorn Loss**: Optimal transport-based sequence alignment
- **Wasserstein Loss**: Earth mover's distance for robust training
- **Custom Validation Metrics**: Advanced evaluation beyond traditional TPP metrics



## Dataset <a href='#top'>[Back to Top]</a>
<span id='dataset'/>

We preprocessed one synthetic and five real world datasets from widely-cited works that contain diverse characteristics in terms of their application domains and temporal statistics:
- Synthetic: a univariate Hawkes process simulated by [Tick](https://github.com/X-DataInitiative/tick) library.
- Retweet ([Zhou, 2013](http://proceedings.mlr.press/v28/zhou13.pdf)): timestamped user retweet events.
- Taxi ([Whong, 2014](https://chriswhong.com/open-data/foil_nyc_taxi/)): timestamped taxi pick-up events.
- StackOverflow ([Leskovec, 2014](https://snap.stanford.edu/data/)): timestamped user badge reward events in StackOverflow.
- Taobao ([Xue et al, 2022](https://arxiv.org/abs/2210.01753)): timestamped user online shopping behavior events in Taobao platform.
- Amazon ([Xue et al, 2022](https://arxiv.org/abs/2210.01753)): timestamped user online shopping behavior events in Amazon platform.

Per users' request, we processed two non-anthropogenic datasets 
- [Earthquake](https://drive.google.com/drive/folders/1ubeIz_CCNjHyuu6-XXD0T-gdOLm12rf4): timestamped earthquake events over the Conterminous U.S from 1996 to 2023, processed from [USGS](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data).
- [Volcano eruption](https://drive.google.com/drive/folders/1KSWbNi8LUwC-dxz1T5sOnd9zwAot95Tp?usp=drive_link): timestamped volcano eruption events over the world in recent hundreds of years, processed from [The Smithsonian Institution](https://volcano.si.edu/).


  All datasets are preprocess to the `Gatech` format dataset widely used for TPP researchers, and saved at [Google Drive](https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7) with a public access.

## Quick Start <a href='#top'>[Back to Top]</a>
<span id='quick-start'/>


### Colab Tutorials

Explore the following tutorials that can be opened directly in Google Colab:

- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ant-research/EasyTemporalPointProcess/blob/main/notebooks/easytpp_1_dataset.ipynb) Tutorial 1: Dataset in EasyTPP.
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ant-research/EasyTemporalPointProcess/blob/main/notebooks/easytpp_2_tfb_wb.ipynb) Tutorial 2: Tensorboard in EasyTPP.

### End-to-end Example

We provide an end-to-end example for users to run a standard TPP model with `EasyTPP`.


### Step 1. Installation

This project uses modern Python packaging with `pyproject.toml` and `uv` for fast dependency management.

#### Prerequisites

- **uv** (replaces pip and venv) - [Install from here](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.8 or higher (uv can install this automatically)

#### Quick Installation

```bash
# Clone the repository
git clone https://github.com/NzoCs/Learning-point-processes.git
cd Learning-point-processes

# Setup with uv (automatically creates virtual environment and installs dependencies)
uv sync --all-extras
```

#### Installation Options

Choose the dependency groups that fit your needs:

```bash
# Core dependencies only
uv sync

# CLI tools (for command-line interface)
uv sync --group cli

# Development tools (testing, linting, formatting)
uv sync --group dev

# Documentation tools
uv sync --group docs

# All optional dependencies
uv sync --all-extras
```

Note: With `uv sync`, the project is automatically installed in editable mode.

#### CLI Interface Setup

The project includes a comprehensive CLI located in the `scripts/` directory:

```bash
# Navigate to scripts directory
cd scripts

# Test CLI installation (uv automatically uses project environment)
uv run python easytpp_cli.py --version
uv run python easytpp_cli.py --help

# Run interactive setup
uv run python easytpp_cli.py interactive

# Display system information
uv run python easytpp_cli.py info
```

#### Development Setup

For development work, install additional tools:

```bash
# Install development dependencies
uv sync --group dev

# Set up pre-commit hooks (optional)
uv run pre-commit install

# Verify installation
uv run python check_installation.py
```

### Step 2. Prepare datasets 

We need to put the datasets in a local directory before running a model and the datasets should follow a certain format. See [OnlineDoc - Datasets](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/dataset.html) for more details.

Suppose we use the [taxi dataset](https://chriswhong.com/open-data/foil_nyc_taxi/) in the example.

### Step 3. Train the model


Before start training, we need to set up the config file for the pipeline. We provide a preset config file in [Example Config](https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/configs/experiment_config.yaml). The details of the configuration can be found in [OnlineDoc - Training Pipeline](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/run_train_pipeline.html).

After the setup of data and config, the directory structure is as follows:

```bash

    data
     |______taxi
             |____ train.pkl
             |____ dev.pkl
             |____ test.pkl

    configs
     |______experiment_config.yaml

```


Then we start the training by running the script:

```python
from pathlib import Path
from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runners import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def main():
    # Load configuration
    config_path = "configs/runner_config.yaml"
    config_dict = parse_runner_yaml_config(config_path, "NHP", "train")
    config = RunnerConfig.from_dict(config_dict)
    
    # Create and run the model
    runner = Runner(config=config)
    runner.run(phase="train")


if __name__ == '__main__':
    main()
```

### Alternative: Complete Pipeline Example

You can also run a complete pipeline (train -> test -> predict) using the example from `examples/run_all_phase.py`:

```python
from pathlib import Path
from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runners import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def main():
    # Load configuration
    config_path = "configs/test_runner_config.yaml"
    config_dict = parse_runner_yaml_config(config_path, "NHP", "test")
    config = RunnerConfig.from_dict(config_dict)
    
    # Create runner
    runner = Runner(config=config)
    
    # Run complete pipeline: train -> test -> predict
    
    # 1. Training
    runner.run(phase="train")
    
    # 2. Testing
    runner.run(phase="test")
    
    # 3. Prediction and distribution comparison
    runner.run(phase="predict")


if __name__ == "__main__":
    main()
```

A more detailed example can be found at [OnlineDoc - QuickStart](https://ant-research.github.io/EasyTemporalPointProcess/get_started/quick_start.html).

### Modern CLI Interface

This project includes a comprehensive CLI interface located in the `scripts/` directory. The CLI provides an intuitive way to run experiments, generate data, and manage configurations.

#### Quick CLI Commands

```bash
# Navigate to scripts directory
cd scripts

# Show all available commands  
uv run python easytpp_cli.py --help

# Show system information
uv run python easytpp_cli.py info

# List available configurations
uv run python easytpp_cli.py list-configs --dir ../configs

# Run interactive mode (recommended for beginners)
uv run python easytpp_cli.py interactive

# Validate a configuration
uv run python easytpp_cli.py validate --config ../configs/runner_config.yaml --experiment THP --dataset H2expc

# Run an experiment
uv run python easytpp_cli.py run --config ../configs/runner_config.yaml --experiment THP --dataset H2expc --phase test
```

#### Advanced CLI Features

The CLI supports comprehensive TPP workflows:

```bash
# Generate synthetic data
uv run python easytpp_cli.py data-gen --type hawkes --num-sims 100 --output ./data/synthetic

# Inspect and visualize data  
uv run python easytpp_cli.py data-inspect --experiment H2expi --output ./visualizations

# Run benchmarks for comparison
uv run python easytpp_cli.py benchmark --type mean --dataset test --output ./benchmark_results

# Train a model with custom parameters
uv run python easytpp_cli.py run \
  --config ../configs/training_config.yaml \
  --experiment THP \
  --dataset taxi \
  --phase train \
  --device gpu \
  --seed 42
```

#### CLI Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `info` | Display system information | `uv run python easytpp_cli.py info` |
| `list-configs` | List configuration files | `uv run python easytpp_cli.py list-configs --dir ../configs` |
| `interactive` | Launch interactive mode | `uv run python easytpp_cli.py interactive` |
| `validate` | Validate configuration | `uv run python easytpp_cli.py validate --config config.yaml --experiment THP` |
| `run` | Run experiment | `uv run python easytpp_cli.py run --config config.yaml --experiment THP --phase test` |
| `data-gen` | Generate synthetic data | `uv run python easytpp_cli.py data-gen --type hawkes --num-sims 100` |
| `data-inspect` | Visualize and analyze data | `uv run python easytpp_cli.py data-inspect --experiment H2expi` |
| `benchmark` | Run performance benchmarks | `uv run python easytpp_cli.py benchmark --type mean --dataset test` |

#### Interactive Mode

For beginners, the interactive mode provides guided setup:

```bash
cd scripts
uv run python easytpp_cli.py interactive
```

This will guide you through:

- Configuration file selection
- Experiment and dataset selection  
- Parameter configuration
- Execution confirmation

#### CLI Parameters

Common parameters for CLI commands:

- `--config, -c`: Path to YAML configuration file
- `--experiment, -e`: Experiment ID (e.g., THP, NHP, SAHP)
- `--dataset, -d`: Dataset ID (e.g., H2expc, taxi, retweet)
- `--phase, -p`: Execution phase (train, test, predict, validation, all)
- `--device`: Computation device (auto, cpu, gpu)
- `--output, -o`: Output directory for results
- `--seed`: Random seed for reproducibility
- `--verbose, -v`: Enable verbose logging

## 📁 Codebase Structure

The New-LTPP framework is organized with a modular architecture that separates core functionality, configuration, examples, and tools:

```text
New-LTPP/
├── 📦 Core Framework
│   ├── pyproject.toml                 # Modern Python packaging configuration
│   ├── uv.lock                        # Lockfile for reproducible builds
│   ├── README.md                      # Main documentation  
│   ├── SETUP_GUIDE.md                # Detailed setup instructions
│   ├── check_installation.py         # Installation verification script
│   └── Makefile                      # Build automation
│
├── 🧠 Core Library (easy_tpp/)
│   ├── config_factory/               # Configuration management system
│   │   ├── __init__.py
│   │   ├── data_config.py           # Data loading configurations
│   │   ├── model_config.py          # Model-specific configurations
│   │   ├── runner_config.py         # Training pipeline configurations
│   │   └── hpo_config.py            # Hyperparameter optimization configs
│   ├── models/                       # TPP model implementations
│   │   ├── __init__.py
│   │   ├── basemodel.py             # Base model interface
│   │   ├── nhp.py                   # Neural Hawkes Process
│   │   ├── thp.py                   # Transformer Hawkes Process
│   │   ├── rmtpp.py                 # Recurrent Marked TPP
│   │   ├── sahp.py                  # Self-Attentive Hawkes Process
│   │   ├── attnhp.py               # Attentive Neural Hawkes Process
│   │   ├── fullynn.py              # Fully Neural Network TPP
│   │   ├── intensity_free.py       # Intensity-Free TPP
│   │   ├── ode_tpp.py              # ODE-based TPP
│   │   ├── hawkes.py               # Classical Hawkes Process
│   │   └── self_correcting.py      # Self-Correcting Process
│   ├── data/                        # Data processing and generation
│   │   ├── generation/              # Synthetic data generation
│   │   └── preprocessing/           # Data preprocessing utilities
│   ├── evaluation/                  # Advanced evaluation metrics
│   │   └── benchmarks/              # Baseline comparison tools
│   ├── runners/                     # Training and execution pipeline
│   │   ├── __init__.py
│   │   ├── runner.py               # Main runner interface
│   │   ├── model_runner.py         # Model training runner
│   │   ├── evaluation_runner.py    # Evaluation runner
│   │   ├── hpo_runner.py           # HPO runner
│   │   └── data_generation_runner.py # Data generation runner
│   ├── hpo/                        # Hyperparameter optimization
│   │   └── hypertuner.py           # Custom hyperparameter tuning
│   └── utils/                       # Utility functions
│       └── yaml_config_utils.py    # YAML configuration utilities
│
├── ⚙️ Configuration Templates (configs/)
│   ├── runner_config.yaml           # Main training configuration
│   ├── test_runner_config.yaml      # Test configuration
│   ├── bench_config.yaml            # Benchmark configuration
│   └── hpo_config.yaml             # HPO configuration template
│
├── 🚀 Command Line Interface (scripts/)
│   ├── easytpp_cli.py              # Main CLI application
│   ├── CLI_README.md               # Detailed CLI documentation
│   ├── run_all_pipeline.sh         # Batch execution script
│   └── train_ruche_cpu.sh          # HPC execution script
│
├── 📚 Examples & Tutorials (examples/)
│   ├── run_all_phase.py            # Complete pipeline example
│   ├── train_nhp_hpo.py            # HPO training example
│   ├── benchmark.py                # Benchmarking examples
│   ├── data_inspection.py          # Data analysis example
│   ├── gen_synthetic_data.py       # Data generation example
│   ├── prediction_analysis.py      # Prediction and analysis
│   ├── data_loader.py              # Data loading utilities
│   └── event_tokenizer.py          # Event tokenization utilities
│
├── 📓 Interactive Tutorials (notebooks/)
│   └── EasyTPP_Getting_Started.ipynb  # Comprehensive tutorial notebook
│
├── 🧪 Test Suite (tests/)
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── functional/                 # Functional tests
│   └── conftest.py                # Test configuration
│
├── 📖 Documentation (docs/)
│   ├── make.bat                    # Windows documentation build
│   ├── Makefile                    # Unix documentation build
│   └── source/                     # Sphinx documentation source
│
├── 🐳 Deployment (docker/)
│   └── ...                        # Docker configurations
│
├── 📊 Results & Outputs
│   ├── artifacts/                  # Training artifacts and checkpoints
│   └── coverage_html/             # Test coverage reports
│
└── 📋 Project Configuration
    ├── .github/                   # GitHub Actions workflows
    ├── .gitignore                 # Git ignore rules
    ├── .coveragerc               # Coverage configuration
    ├── pytest.ini               # Pytest configuration
    ├── pyproject.toml            # Modern Python packaging
    └── Makefile                  # Build automation
```

### 📋 Typical Data Structure

When working with New-LTPP, your data should be organized as follows:

```text
data/
├── taxi/                          # Dataset name
│   ├── train.pkl                  # Training data (pickle format)
│   ├── dev.pkl                    # Development/validation data  
│   ├── test.pkl                   # Test data
│   └── metadata.json             # Dataset metadata (optional)
├── retweet/                       # Another dataset
│   ├── train.pkl
│   ├── dev.pkl
│   └── test.pkl
└── synthetic/                     # Generated synthetic data
    ├── hawkes_sim/
    │   ├── train.pkl
    │   ├── dev.pkl
    │   └── test.pkl
    └── generated_metadata.json
```

### 🎯 Configuration Structure

Configuration files follow a hierarchical structure:

```text
configs/
├── runner_config.yaml             # Main configuration template
│   ├── pipeline_config_id         # Configuration identifier
│   ├── data_config                # Data loading settings
│   │   ├── train_dir              # Training data directory
│   │   ├── valid_dir              # Validation data directory  
│   │   ├── test_dir               # Test data directory
│   │   ├── data_format            # Data format (json/pickle)
│   │   └── data_loading_specs     # Batch size, workers, etc.
│   ├── model_config               # Model-specific settings
│   │   ├── model_id               # Model type (NHP, THP, etc.)
│   │   ├── hidden_size            # Model dimensions
│   │   ├── num_layers             # Network depth
│   │   └── model_specs            # Model-specific parameters
│   ├── training_config             # Training configuration
│   │   ├── max_epochs             # Training epochs
│   │   ├── learning_rate          # Learning rate
│   │   ├── batch_size             # Training batch size
│   │   └── optimizer_specs        # Optimizer settings
│   └── logger_config              # Logging configuration
├── bench_config.yaml              # Benchmark settings
└── hpo_config.yaml               # HPO configuration
```

### Key Directories

- **`easy_tpp/`**: Core library with model implementations and utilities
- **`scripts/`**: Command-line interface and automation tools  
- **`configs/`**: Configuration templates and examples
- **`examples/`**: Practical examples and tutorials for different use cases
- **`tests/`**: Comprehensive test suite
- **`docs/`**: Documentation source files

For detailed CLI documentation, see [`scripts/CLI_README.md`](scripts/CLI_README.md).


## Documentation <a href='#top'>[Back to Top]</a>
<span id='doc'/>

The classes and methods of `EasyTPP` have been well documented so that users can generate the documentation by:

```shell
cd doc
pip install -r requirements.txt
make html
```
NOTE:
* The `doc/requirements.txt` is only for documentation by Sphinx, which can be automatically generated by Github actions `.github/workflows/docs.yml`. (Trigger by pull request.)

The full documentation is available on the [website](https://ant-research.github.io/EasyTemporalPointProcess/).
 
## Benchmark <a href='#top'>[Back to Top]</a>
<span id='benchmark'/>

In the [examples](https://github.com/ant-research/EasyTemporalPointProcess/tree/main/examples) folder, we provide a [script](https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/benchmark_script.py) to benchmark the TPPs, with Taxi dataset as the input. 

To run the script, one should download the Taxi data following the above instructions. The [config](https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/configs/experiment_config.yaml) file is readily setup up. Then run


```shell
cd examples
python run_retweet.py
```


## License <a href='#top'>[Back to Top]</a>

This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/EasyNLP/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/ant-research/EasyTPP/blob/master/NOTICE) file for more information.


## Todo List

<span id='todo'/>

### 🔥 Upcoming Features

#### Advanced Loss Functions & Metrics

- [ ] **Optimal Transport Losses**:
  - [ ] Gromov-Wasserstein distance implementation
  - [ ] Fused optimal transport for multivariate sequences
- [ ] **Information-Theoretic Metrics**:
  - [ ] Mutual information-based evaluation
  - [ ] KL divergence variants for temporal distributions
- [ ] **Geometric Deep Learning**:
  - [ ] Graph neural network integration for event dependencies
  - [ ] Manifold learning for temporal embeddings

#### Enhanced Model Capabilities

- [ ] **Multi-scale Temporal Modeling**:
  - [ ] Hierarchical attention mechanisms
  - [ ] Wavelet-based temporal decomposition
- [ ] **Causal Discovery**:
  - [ ] Granger causality integration
  - [ ] Causal inference for event relationships
- [ ] **Meta-Learning**:
  - [ ] Few-shot adaptation for new event types
  - [ ] Transfer learning across domains

#### Advanced Evaluation & Analysis

- [ ] **Distributional Testing**:
  - [ ] Kolmogorov-Smirnov tests for generated sequences
  - [ ] Anderson-Darling goodness-of-fit tests
- [ ] **Temporal Pattern Mining**:
  - [ ] Automatic pattern discovery in generated sequences
  - [ ] Burst detection and analysis
- [ ] **Uncertainty Quantification**:
  - [ ] Conformal prediction intervals
  - [ ] Bayesian neural network integration

#### Performance & Scalability

- [ ] **Distributed Training**:
  - [ ] Multi-GPU scaling improvements
  - [ ] Distributed data parallel optimization
- [ ] **Model Compression**:
  - [ ] Knowledge distillation for TPP models
  - [ ] Pruning and quantization techniques
- [ ] **Streaming Inference**:
  - [ ] Real-time prediction capabilities
  - [ ] Online learning adaptation

### 🛠️ Technical Improvements

- [ ] **Enhanced CLI**:
  - [ ] Configuration templates for common use cases
  - [ ] Automated hyperparameter suggestions
- [ ] **Documentation**:
  - [ ] Comprehensive tutorials for advanced features
  - [ ] Best practices guide for evaluation metrics
- [ ] **Testing & Quality**:
  - [ ] Comprehensive unit test coverage
  - [ ] Integration tests for all loss functions
  - [ ] Performance benchmarking suite

### 📊 Research Extensions

- [ ] **New Datasets**:
  - [ ] Financial market events integration
  - [ ] Social media temporal patterns
  - [ ] IoT sensor event sequences
- [ ] **Benchmark Studies**:
  - [ ] Comprehensive comparison with traditional metrics
  - [ ] Robustness analysis under distribution shift
  - [ ] Computational efficiency comparisons

### 🔬 Experimental Features

- [ ] **Generative Modeling**:
  - [ ] VAE integration for temporal point processes
  - [ ] GAN-based sequence generation
- [ ] **Reinforcement Learning**:
  - [ ] RL-based sequence optimization
  - [ ] Multi-agent temporal modeling

## 🤝 Contributing

<span id='contributing'/>

We welcome contributions from the community! This project follows modern development practices and coding standards.

### Quick Start for Contributors

1. **Read the [Contributing Guide](CONTRIBUTING.md)** - Complete guide for contributors
2. **Check [Open Issues](https://github.com/NzoCs/Learning-point-processes/issues)** - Find something to work on
3. **Follow our [Git Workflow](.github/README.md)** - Standardized development process

### Development Setup

```bash
# Clone and setup
git clone https://github.com/NzoCs/Learning-point-processes.git
cd Learning-point-processes
uv sync

# Install pre-commit hooks
pre-commit install

# Run tests
make test
```

### Contribution Types

- 🐛 **Bug Fixes**: Help us maintain reliability
- ✨ **New Features**: Extend the framework capabilities  
- 📚 **Documentation**: Improve clarity and examples
- ⚡ **Performance**: Optimize existing implementations
- 🧪 **Testing**: Improve test coverage and quality

### Standards

- **Code Style**: Black, isort, flake8
- **Commits**: [Conventional Commits](https://conventionalcommits.org/)
- **Testing**: Minimum 90% coverage for new code
- **Documentation**: Google-style docstrings

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Citation

<span id='citation'/>

### Original EasyTPP Citation

This project is greatly inspired by the original EasyTPP framework. If you use this work, please cite the original paper:

```bibtex
@inproceedings{xue2024easytpp,
      title={EasyTPP: Towards Open Benchmarking Temporal Point Processes}, 
      author={Siqiao Xue and Xiaoming Shi and Zhixuan Chu and Yan Wang and Hongyan Hao and Fan Zhou and Caigao Jiang and Chen Pan and James Y. Zhang and Qingsong Wen and Jun Zhou and Hongyuan Mei},
      booktitle = {International Conference on Learning Representations (ICLR)},
      year = {2024},
      url ={https://arxiv.org/abs/2307.08097}
}
```

### This Project

If you find New-LTPP useful for your research, please consider citing this repository:

```bibtex
@software{new_ltpp2025,
      title={New-LTPP: Advanced Temporal Point Process Framework},
      author={Enzo Cao},
      year={2025},
      url={https://github.com/NzoCs/Learning-point-processes},
      note={Advanced TPP framework with Lightning integration and enhanced evaluation metrics}
}
```

## Acknowledgment

<span id='acknowledgment'/>

### Original EasyTPP Team

This project builds upon the excellent foundation provided by the [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess) team from Machine Intelligence Group, Alipay and DAMO Academy, Alibaba. We are grateful for their pioneering work in making TPP research more accessible.

### Key Inspirations

The following repositories and frameworks have influenced this work:

- **[EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)**: Original foundation and inspiration
- **[PyTorch Lightning](https://lightning.ai/)**: High-performance training framework
- **[Neural Hawkes Process](https://github.com/hongyuanmei/neurawkes)**: Fundamental TPP implementations
- **[Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)**: Attention mechanisms in TPP

### Advanced Metrics & Losses

Special thanks to the research community for developing the advanced evaluation metrics and loss functions implemented in this framework, including contributions from optimal transport, distributional analysis, and robust machine learning communities.

