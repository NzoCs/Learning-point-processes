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

| [Features](#features) | [Project Setup](#project-setup) | [Model List](#model-list) | [Dataset](#dataset) | [Quick Start](#quick-start) | [Benchmark](#benchmark) | [Documentation](#doc) | [Todo List](#todo) | [Citation](#citation) | [Acknowledgment](#acknowledgment) |

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

This project uses modern Python packaging with `pyproject.toml` and includes a comprehensive CLI interface located in the `scripts/` directory.

### Prerequisites

- Python 3.8 or higher
- pip 21.3+ (for full pyproject.toml support)
- Git

### Quick Setup

```bash
# Clone this repository
git clone https://github.com/NzoCs/Learning-point-processes.git
cd Learning-point-processes

# Create and activate virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[all]"
```

### Installation Options

Choose the installation that fits your needs:

```bash
# Minimal installation (core dependencies only)
pip install -e .

# Development installation (includes testing, linting, formatting tools)
pip install -e ".[dev]"

# CLI tools installation (for command-line usage)
pip install -e ".[cli]"

# Documentation tools (for building docs)
pip install -e ".[docs]"

# Everything (all optional dependencies)
pip install -e ".[all]"
```

### CLI Interface

The project includes a comprehensive CLI interface located in the `scripts/` directory. After installation, you can access the CLI:

```bash
# Navigate to the scripts directory
cd scripts

# Run the main CLI
python easytpp_cli.py --help

# Quick installation verification
python easytpp_cli.py info

# Interactive mode for guided setup
python easytpp_cli.py interactive
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
python check_installation.py

# Test the CLI interface
cd scripts
python easytpp_cli.py --version
python easytpp_cli.py info
```


## Model List
<span id='model-list'/>

New-LTPP implements state-of-the-art TPP models with Lightning integration and enhanced loss functions:

| No  | Publication |     Model     | Paper                                                                                                                                    | Lightning Implementation                                                                                                   |
|:---:|:-----------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
|  1  |   KDD'16    |     RMTPP     | [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) | [Lightning Module](easy_tpp/model/torch_model/torch_rmtpp.py)                   |
|  2  | NeurIPS'17  |      NHP      | [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)                     | [Lightning Module](easy_tpp/model/torch_model/torch_nhp.py)                       |
|  3  | NeurIPS'19  |    FullyNN    | [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690)                                | [Lightning Module](easy_tpp/model/torch_model/torch_fullynn.py)                |
|  4  |   ICML'20   |     SAHP      | [Self-Attentive Hawkes process](https://arxiv.org/abs/1907.07561)                                                                        | [Lightning Module](easy_tpp/model/torch_model/torch_sahp.py)                     |
|  5  |   ICML'20   |      THP      | [Transformer Hawkes process](https://arxiv.org/abs/2002.09291)                                                                           | [Lightning Module](easy_tpp/model/torch_model/torch_thp.py)                       |
|  6  |   ICLR'20   | IntensityFree | [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127)                                                  | [Lightning Module](easy_tpp/model/torch_model/torch_intensity_free.py) |
|  7  |   ICLR'21   |    ODETPP     | [Neural Spatio-Temporal Point Processes (simplified)](https://arxiv.org/abs/2011.04583)                                                  | [Lightning Module](easy_tpp/model/torch_model/torch_ode_tpp.py)               |
|  8  |   ICLR'22   |    AttNHP     | [Transformer Embeddings of Irregularly Spaced Events and Their Participants](https://arxiv.org/abs/2201.00044)                           | [Lightning Module](easy_tpp/model/torch_model/torch_attnhp.py)                 |

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

This project uses modern Python packaging with `pyproject.toml` for simplified dependency management.

#### Prerequisites

- Python 3.8 or higher  
- pip 21.3+ (for full pyproject.toml support)

#### Quick Installation

```bash
# Clone the repository
git clone https://github.com/NzoCs/Learning-point-processes.git
cd Learning-point-processes

# Create virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install with all features
pip install -e ".[all]"
```

#### Installation Options

Choose the installation that fits your needs:

```bash
# Core dependencies only
pip install -e .

# CLI tools (for command-line interface)
pip install -e ".[cli]"

# Development tools (testing, linting, formatting)
pip install -e ".[dev]"

# Documentation tools
pip install -e ".[docs]"

# All optional dependencies
pip install -e ".[all]"
```

#### CLI Interface Setup

The project includes a comprehensive CLI located in the `scripts/` directory:

```bash
# Navigate to scripts directory
cd scripts

# Test CLI installation
python easytpp_cli.py --version
python easytpp_cli.py --help

# Run interactive setup
python easytpp_cli.py interactive

# Display system information
python easytpp_cli.py info
```

#### Development Setup

For development work, install additional tools:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (optional)
pre-commit install

# Verify installation
python check_installation.py
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


Then we start the training by simply running the script 

```python

import argparse
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='NHP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

    model_runner = Runner.build_from_config(config)

    model_runner.run()


if __name__ == '__main__':
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
python easytpp_cli.py --help

# Show system information
python easytpp_cli.py info

# List available configurations
python easytpp_cli.py list-configs --dir ../configs

# Run interactive mode (recommended for beginners)
python easytpp_cli.py interactive

# Validate a configuration
python easytpp_cli.py validate --config ../configs/runner_config.yaml --experiment THP --dataset H2expc

# Run an experiment
python easytpp_cli.py run --config ../configs/runner_config.yaml --experiment THP --dataset H2expc --phase test
```

#### Advanced CLI Features

The CLI supports comprehensive TPP workflows:

```bash
# Generate synthetic data
python easytpp_cli.py data-gen --type hawkes --num-sims 100 --output ./data/synthetic

# Inspect and visualize data  
python easytpp_cli.py data-inspect --experiment H2expi --output ./visualizations

# Run benchmarks for comparison
python easytpp_cli.py benchmark --type mean --dataset test --output ./benchmark_results

# Train a model with custom parameters
python easytpp_cli.py run \
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
| `info` | Display system information | `python easytpp_cli.py info` |
| `list-configs` | List configuration files | `python easytpp_cli.py list-configs --dir ../configs` |
| `interactive` | Launch interactive mode | `python easytpp_cli.py interactive` |
| `validate` | Validate configuration | `python easytpp_cli.py validate --config config.yaml --experiment THP` |
| `run` | Run experiment | `python easytpp_cli.py run --config config.yaml --experiment THP --phase test` |
| `data-gen` | Generate synthetic data | `python easytpp_cli.py data-gen --type hawkes --num-sims 100` |
| `data-inspect` | Visualize and analyze data | `python easytpp_cli.py data-inspect --experiment H2expi` |
| `benchmark` | Run performance benchmarks | `python easytpp_cli.py benchmark --type mean --dataset test` |

#### Interactive Mode

For beginners, the interactive mode provides guided setup:

```bash
cd scripts
python easytpp_cli.py interactive
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
│   ├── README.md                      # Main documentation  
│   ├── SETUP_GUIDE.md                # Detailed setup instructions
│   ├── check_installation.py         # Installation verification script
│   └── Makefile                      # Build automation
│
├── 🧠 Core Library (easy_tpp/)
│   ├── config_factory/               # Configuration management system
│   │   ├── __init__.py
│   │   ├── base.py                   # Base configuration classes
│   │   ├── data_config.py           # Data loading configurations
│   │   ├── model_config.py          # Model-specific configurations
│   │   ├── runner_config.py         # Training pipeline configurations
│   │   ├── logger_config.py         # Logging configurations
│   │   └── hpo_config.py            # Hyperparameter optimization configs
│   ├── models/                       # TPP model implementations
│   │   ├── __init__.py
│   │   ├── basemodel.py             # Base model interface
│   │   ├── nhp.py                   # Neural Hawkes Process
│   │   ├── thp.py                   # Transformer Hawkes Process
│   │   ├── rmtpp.py                 # Recurrent Marked TPP
│   │   ├── sahp.py                  # Self-Attentive Hawkes Process
│   │   ├── attnhp.py               # Attentive Neural Hawkes Process
│   │   └── ...                      # Additional model implementations
│   ├── data/                        # Data processing and generation
│   │   ├── preprocess/              # Data preprocessing utilities
│   │   └── generation/              # Synthetic data generation
│   ├── evaluation/                  # Advanced evaluation metrics
│   │   ├── benchmarks/              # Baseline comparison tools
│   │   ├── metrics/                 # Custom evaluation metrics
│   │   └── distribution_analysis/   # Temporal distribution analysis
│   ├── runner/                      # Training and execution pipeline
│   │   ├── __init__.py
│   │   ├── base_runner.py          # Base runner interface
│   │   └── lightning_runner.py     # PyTorch Lightning integration
│   ├── hpo/                        # Hyperparameter optimization
│   │   ├── __init__.py
│   │   ├── optuna_tuner.py         # Optuna-based optimization
│   │   └── hypertuner.py           # Custom hyperparameter tuning
│   └── utils/                       # Utility functions
│       ├── torch_utils.py          # PyTorch utilities
│       ├── device_utils.py         # Device management
│       ├── generic.py              # Generic helper functions
│       └── ...                     # Additional utilities
│
├── ⚙️ Configuration Templates (configs/)
│   ├── runner_config.yaml           # Main training configuration
│   ├── bench_config.yaml            # Benchmark configuration
│   └── hpo_config.yaml             # HPO configuration template
│
├── 🚀 Command Line Interface (scripts/)
│   ├── easytpp_cli.py              # Main CLI application
│   ├── CLI_README.md               # Detailed CLI documentation
│   └── setup_cli.py                # CLI setup and utilities
│
├── 🎯 Execution Workflows (main/)
│   ├── data_gen/                   # Data generation workflows
│   │   ├── run_gen.py              # Generate synthetic data
│   │   └── simple_data_gen.py      # Simple generation examples
│   ├── data_inspection/            # Data analysis workflows
│   │   ├── run_insp.py             # Data inspection pipeline
│   │   ├── simple_data_inspection.py
│   │   └── config.yaml             # Inspection configuration
│   ├── run_benchmarks/             # Benchmark execution
│   │   ├── run_bench.py            # Benchmark runner
│   │   ├── simple_benchmark.py     # Simple benchmark examples
│   │   ├── bench_config.yaml       # Benchmark settings
│   │   └── README.md               # Benchmark documentation
│   └── run_model/                  # Model training workflows
│       ├── run_model.py            # Main model runner
│       ├── train_example.py        # Training examples
│       ├── minimal_example.py      # Minimal usage example
│       ├── runner_config.yaml      # Model training configuration
│       ├── run_all_pipeline.sh     # Batch execution script
│       └── train_ruche_cpu.sh      # HPC execution script
│
├── 📚 Examples & Tutorials (examples/)
│   ├── simple_example.py           # Basic usage example
│   ├── prediction_analysis.py      # Prediction and analysis
│   ├── train_nhp.py               # NHP training example
│   ├── train_nhp_hpo.py           # HPO example
│   ├── benchmark.py               # Benchmarking example
│   ├── data_inspection.py         # Data analysis example
│   ├── gen_synthetic_data.py      # Data generation example
│   ├── hf_data_loader.py          # HuggingFace data loading
│   ├── runner_config.yaml         # Example configuration
│   ├── script_data_processing/    # Data processing scripts
│   │   ├── taxi.py                # Taxi dataset processing
│   │   ├── earthquake.py          # Earthquake data processing
│   │   ├── volcano.py             # Volcano data processing
│   │   ├── taobao.py             # Taobao dataset processing
│   │   └── make_hf_dataset.py     # HuggingFace dataset creation
│   └── train_experiment/          # Training experiments
│       ├── run_retweet.py         # Retweet dataset experiment
│       └── retweet_config.yaml    # Retweet configuration
│
├── 📓 Interactive Tutorials (notebooks/)
│   └── EasyTPP_Getting_Started.ipynb  # Comprehensive tutorial notebook
│
├── 🧪 Test Suite (tests/)
│   ├── unit/                       # Unit tests
│   │   ├── config/                 # Configuration tests
│   │   ├── models/                 # Model tests
│   │   ├── preprocess/             # Data processing tests
│   │   ├── runner/                 # Runner tests
│   │   └── utils/                  # Utility tests
│   ├── integration/                # Integration tests
│   ├── functional/                 # Functional tests
│   ├── benchmarks/                 # Benchmark tests
│   ├── device/                     # Device consistency tests
│   ├── conftest.py                # Test configuration
│   └── test_cli.py                # CLI tests
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
│   ├── checkpoints/               # Model checkpoints
│   │   ├── NHP/                   # NHP model checkpoints
│   │   └── THP/                   # THP model checkpoints
│   ├── lightning_logs/            # PyTorch Lightning logs
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
│   ├── trainer_config             # Training configuration
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
- **`main/`**: High-level execution scripts for different workflows
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
- **[Neural Hawkes Particle Smoothing](https://github.com/hongyuanmei/neural-hawkes-particle-smoothing)**: Advanced TPP techniques
- **[Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)**: Attention mechanisms in TPP

### Advanced Metrics & Losses

Special thanks to the research community for developing the advanced evaluation metrics and loss functions implemented in this framework, including contributions from optimal transport, distributional analysis, and robust machine learning communities.

