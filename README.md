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

**New-LTPP** is a modern, advanced framework for [Temporal Point Process](https://mathworld.wolfram.com/TemporalPointProcess.html) (TPP) research and development. Originally inspired by [new_ltpp](https://github.com/ant-research/EasyTemporalPointProcess), this project has evolved into a comprehensive toolkit with significant enhancements in performance, usability, and research capabilities.

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
- **[2025-07]** 🎯 **Enhanced Evaluation**: Comprehensive validation metrics including distributional analysis
- **[2025-07]** 🛠️ **Modern CLI**: Professional command-line interface with interactive modes
- **[2025-07]** ⚡ **Performance Boost**: Significant training speed improvements through Lightning optimization
- **[2025-07]** 🔬 **Simulation Capabilities**: Built-in model simulation and temporal pattern analysis


## Project Setup
<span id='project-setup'/>

For installation and environment setup, see the dedicated Setup Guide: `SETUP.md`.

### Makefile workflow (recommended)

Use the provided Makefile to manage common tasks:

```bash
# Show all available commands
make help

# Sync dependencies and install everything
make install-all

# Run tests and quality checks
make test
make quality

# Run quick examples
make run-nhp     # NHP on test dataset
make run-thp     # THP on test dataset
make benchmark-all
```

Key targets: `help`, `install-all`, `uv-sync`, `test`, `run-nhp`, `run-thp`, `benchmark`, `docs`.


## Model List
<span id='model-list'/>

New-LTPP implements state-of-the-art TPP models with modern PyTorch implementations and enhanced evaluation capabilities:

| No  | Publication |     Model     | Paper                                                                                                                                    | Implementation                                                                                                   |
|:---:|:-----------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
|  1  |   KDD'16    |     RMTPP     | [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) | [Model](new_ltpp/models/rmtpp.py)                   |
|  2  | NeurIPS'17  |      NHP      | [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)                     | [Model](new_ltpp/models/nhp.py)                       |
|  3  | NeurIPS'19  |    FullyNN    | [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690)                                | [Model](new_ltpp/models/fullynn.py)                |
|  4  |   ICML'20   |     SAHP      | [Self-Attentive Hawkes process](https://arxiv.org/abs/1907.07561)                                                                        | [Model](new_ltpp/models/sahp.py)                     |
|  5  |   ICML'20   |      THP      | [Transformer Hawkes process](https://arxiv.org/abs/2002.09291)                                                                           | [Model](new_ltpp/models/thp.py)                       |
|  6  |   ICLR'20   | IntensityFree | [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127)                                                  | [Model](new_ltpp/models/intensity_free.py) |
|  7  |   ICLR'21   |    ODETPP     | [Neural Spatio-Temporal Point Processes (simplified)](https://arxiv.org/abs/2011.04583)                                                  | [Model](new_ltpp/models/ode_tpp.py)               |
|  8  |   ICLR'22   |    AttNHP     | [Transformer Embeddings of Irregularly Spaced Events and Their Participants](https://arxiv.org/abs/2201.00044)                           | [Model](new_ltpp/models/attnhp.py)                 |
|  9  |   Custom    |    Hawkes     | Classical Hawkes Process implementation                                                                                                     | [Model](new_ltpp/models/hawkes.py)                |
| 10  |   Custom    | SelfCorrect   | Self-Correcting Point Process                                                                                                               | [Model](new_ltpp/models/self_correcting.py)       |



## Dataset <a href='#top'>[Back to Top]</a>
<span id='dataset'/>

### Available Datasets

This framework supports datasets from two main sources:

#### 1. EasyTPP Real-World Datasets

Preprocessed datasets from [EasyTPP](https://huggingface.co/easytpp) available on Hugging Face, including:

- **Retweet** ([Zhou, 2013](http://proceedings.mlr.press/v28/zhou13.pdf)): Timestamped user retweet events
- **Taxi** ([Whong, 2014](https://chriswhong.com/open-data/foil_nyc_taxi/)): Timestamped taxi pick-up events
- **StackOverflow** ([Leskovec, 2014](https://snap.stanford.edu/data/)): Timestamped user badge reward events
- **Taobao** ([Xue et al, 2022](https://arxiv.org/abs/2210.01753)): Timestamped user online shopping behavior events
- **Amazon** ([Xue et al, 2022](https://arxiv.org/abs/2210.01753)): Timestamped user online shopping behavior events

#### 2. Synthetic Datasets

Custom synthetic datasets generated for TPP research, available on [Hugging Face](https://huggingface.co/NzoCs):

- **Hawkes Process Simulations**: Various parameter configurations
- **Self-Correcting Process**: Different excitation/inhibition patterns
- **Neural Hawkes Process**: Simulated event sequences
- **Custom TPP Models**: Additional synthetic variations

All datasets are preprocessed to the standard format used by TPP researchers and are publicly accessible through Hugging Face.



## Quick Start <a href='#top'>[Back to Top]</a>
<span id='quick-start'/>

See `SETUP.md` for installation. Below are concise ways to run and use the framework.

### Run with Makefile (recommended)

```bash
# Quick runs
make run-nhp      # Train NHP on the test dataset (quick settings)
make run-thp      # Train THP on the test dataset

# Full pipeline
make full-pipeline   # train -> test -> predict

# Benchmarks and inspection
make benchmark-all
make inspect DIR=./data/test
```

### Python API example (YAML-based)

```python
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "yaml_configs" / "configs.yaml"

from new_ltpp.configs import ConfigFactory, ConfigType
from new_ltpp.configs.config_builder import RunnerConfigBuilder
from new_ltpp.runners import RunnerManager


def main() -> None:
  # Load configuration
  config_path = CONFIGS_DIR
  model_id = "NHP"

  # Build runner configuration from YAML
  config_builder = RunnerConfigBuilder()

  # You can modify the paths below to point to different configurations as needed
  config_builder.load_from_yaml(
    yaml_file_path=config_path,
    data_config_path="data_configs.test",
    training_config_path="training_configs.quick_test",
    model_config_path="model_configs.neural_small",
    thinning_config_path="thinning_configs.thinning_fast",
    simulation_config_path="simulation_configs.simulation_fast",
    data_loading_config_path="data_loading_configs.quick_test",
    logger_config_path="logger_configs.mlflow",
  )

  config = config_builder.build()

  # Create runner
  runner = RunnerManager(config=config)

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

More end-to-end scripts are available in the `examples/` folder, e.g.:

- `examples/run_all_phase.py`: complete pipeline (train/test/predict)
- `examples/benchmark_manger_example.py`: run reference benchmarks
- `examples/gen_synthetic_data.py`: synthetic data generation
- `examples/data_inspection.py`: dataset inspection and analysis
- `examples/train_nhp_hpo.py`: HPO training example for NHP
- `examples/event_tokenizer.py`: event tokenization utilities
- `examples/data_loader.py`: data loading utilities

### Modern CLI Interface

This project includes a comprehensive CLI interface located in the `scripts/` directory. The CLI provides an intuitive way to run experiments, generate data, and manage configurations.

#### Quick CLI Commands

```bash
# Navigate to scripts directory
cd scripts

# Show all available commands  
uv run python new_ltpp_cli.py --help

# Show system information
uv run python new_ltpp_cli.py info

# List available configurations
uv run python new_ltpp_cli.py list-configs --dir ../configs

# Run interactive mode (recommended for beginners)
uv run python new_ltpp_cli.py interactive

# Validate a configuration
uv run python new_ltpp_cli.py validate --config ../configs/runner_config.yaml --experiment THP --dataset H2expc

# Run an experiment
uv run python new_ltpp_cli.py run --config ../configs/runner_config.yaml --experiment THP --dataset H2expc --phase test
```

#### Advanced CLI Features

The CLI supports comprehensive TPP workflows:

```bash
# Generate synthetic data
uv run python new_ltpp_cli.py data-gen --type hawkes --num-sims 100 --output ./data/synthetic

# Inspect and visualize data  
uv run python new_ltpp_cli.py data-inspect --experiment H2expi --output ./visualizations

# Run benchmarks for comparison
uv run python new_ltpp_cli.py benchmark --type mean --dataset test --output ./benchmark_results

# Train a model with custom parameters
uv run python new_ltpp_cli.py run \
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
| `info` | Display system information | `uv run python new_ltpp_cli.py info` |
| `list-configs` | List configuration files | `uv run python new_ltpp_cli.py list-configs --dir ../configs` |
| `interactive` | Launch interactive mode | `uv run python new_ltpp_cli.py interactive` |
| `validate` | Validate configuration | `uv run python new_ltpp_cli.py validate --config config.yaml --experiment THP` |
| `run` | Run experiment | `uv run python new_ltpp_cli.py run --config config.yaml --experiment THP --phase test` |
| `data-gen` | Generate synthetic data | `uv run python new_ltpp_cli.py data-gen --type hawkes --num-sims 100` |
| `data-inspect` | Visualize and analyze data | `uv run python new_ltpp_cli.py data-inspect --experiment H2expi` |
| `benchmark` | Run performance benchmarks | `uv run python new_ltpp_cli.py benchmark --type mean --dataset test` |

#### Interactive Mode

For beginners, the interactive mode provides guided setup:

```bash
cd scripts
uv run python new_ltpp_cli.py interactive
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

## Create Custom Models

You can add your own TPP model by subclassing `new_ltpp.models.basemodel.Model` and implementing the required abstract methods:

- `loglike_loss(batch) -> (loss, num_events)`
- `compute_intensities_at_sample_times(time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs)`

Optional hooks you may leverage:

- `predict_one_step(...)` and `predict_one_step_at_every_event(...)`
- `configure_optimizers()` if you need a custom optimizer/scheduler

Register your model in the model registry (see existing models under `new_ltpp/models/`) and ensure it accepts a `ModelConfig` in the constructor. Check the examples in `examples/` for guidance on wiring your model into the runner.

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
├── 🧠 Core Library (new_ltpp/)
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
│   ├── new_ltpp_cli.py              # Main CLI application
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
│   └── new_ltpp_Getting_Started.ipynb  # Comprehensive tutorial notebook
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

- **`new_ltpp/`**: Core library with model implementations and utilities
- **`scripts/`**: Command-line interface and automation tools  
- **`configs/`**: Configuration templates and examples
- **`examples/`**: Practical examples and tutorials for different use cases
- **`tests/`**: Comprehensive test suite
<!-- - **`docs/`**: Documentation source files -->

For detailed CLI documentation, see [`scripts/CLI_README.md`](scripts/CLI_README.md).

<!-- 
## Documentation <a href='#top'>[Back to Top]</a>
<span id='doc'/>

The classes and methods of `new_ltpp` have been well documented so that users can generate the documentation by:

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
``` -->
