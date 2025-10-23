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
- **`docs/`**: Documentation source files

For detailed CLI documentation, see [`scripts/CLI_README.md`](scripts/CLI_README.md).


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
```


## License <a href='#top'>[Back to Top]</a>

This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/EasyNLP/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/ant-research/new_ltpp/blob/master/NOTICE) file for more information.


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

### Original new_ltpp Citation

This project is greatly inspired by the original new_ltpp framework. If you use this work, please cite the original paper:

```bibtex
@inproceedings{xue2024new_ltpp,
      title={new_ltpp: Towards Open Benchmarking Temporal Point Processes}, 
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

### Original new_ltpp Team

This project builds upon the excellent foundation provided by the [new_ltpp](https://github.com/ant-research/EasyTemporalPointProcess) team from Machine Intelligence Group, Alipay and DAMO Academy, Alibaba. We are grateful for their pioneering work in making TPP research more accessible.

### Key Inspirations

The following repositories and frameworks have influenced this work:

- **[easytpp](https://github.com/ant-research/EasyTemporalPointProcess)**: Original foundation and inspiration
- **[PyTorch Lightning](https://lightning.ai/)**: High-performance training framework
- **[Neural Hawkes Process](https://github.com/hongyuanmei/neurawkes)**: Fundamental TPP implementations
- **[Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)**: Attention mechanisms in TPP

### Advanced Metrics & Losses

Special thanks to the research community for developing the advanced evaluation metrics and loss functions implemented in this framework, including contributions from optimal transport, distributional analysis, and robust machine learning communities.

