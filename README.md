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

This project uses modern Python packaging with `pyproject.toml` and PyTorch Lightning for enhanced performance.

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

### Development Tools

The project includes pre-configured development tools:

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

First of all, we can install the package either by using pip or from the source code on Github.

To install the latest stable version:
```bash
pip install easy-tpp
```

To install from the source code (modern pyproject.toml setup):
```bash
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
pip install -e .
```

#### Development Installation

For development, install with additional dependencies:

```bash
# Install with all development tools (recommended for contributors)
pip install -e ".[dev]"

# Or install specific dependency groups:
pip install -e ".[cli]"      # CLI tools
pip install -e ".[docs]"     # Documentation tools  
pip install -e ".[all]"      # All optional dependencies
```

#### Available Dependency Groups

- **Base installation**: Core dependencies only
- **`cli`**: Command-line interface tools (rich, typer, etc.)
- **`docs`**: Documentation generation (sphinx, themes, etc.)
- **`dev`**: Development tools (pytest, black, flake8, mypy, pre-commit)
- **`all`**: All optional dependencies combined

#### Python Environment Setup

This project uses modern Python packaging with `pyproject.toml`. Requirements:

- Python 3.8 or higher
- pip 21.3+ (for full pyproject.toml support)

Create a virtual environment (recommended):

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the package
pip install -e ".[dev]"

# Verify installation
python check_installation.py
```

### Verification

After installation, run the verification script to ensure everything is working:

```bash
python check_installation.py
```

This script will check:
- Python version compatibility
- Core dependencies installation
- Optional dependencies availability
- Import functionality

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

This project now includes a modern CLI interface integrated into the Makefile for easy access:

#### Quick CLI Commands

```bash
# Show all available commands
make help

# Show system information
make info

# List available configurations
make configs

# Run interactive mode
make interactive

# Validate a configuration
make validate CONFIG=configs/examples_runner_config.yaml EXP=THP DATASET=H2expc

# Run an experiment
make run CONFIG=configs/examples_runner_config.yaml EXP=THP DATASET=H2expc PHASE=test
```

#### CLI Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `make cli-info` | Display system information | `make cli-info` |
| `make cli-list-configs` | List configuration files | `make cli-list-configs DIR=./configs` |
| `make cli-interactive` | Launch interactive mode | `make cli-interactive` |
| `make cli-validate` | Validate configuration | `make cli-validate CONFIG=config.yaml EXP=THP DATASET=H2expc` |
| `make cli-run` | Run experiment | `make cli-run CONFIG=config.yaml EXP=THP DATASET=H2expc PHASE=test` |

#### Advanced Usage

```bash
# Run with custom parameters
make cli-run CONFIG=configs/examples_runner_config.yaml EXP=THP DATASET=H2expc PHASE=train DEVICE=cpu VERBOSE=1

# Quick system test
make cli-quick-test

# Pre-defined examples
make cli-example-thp    # THP model example
make cli-example-nhp    # NHP model example
```

#### Parameters

- `CONFIG`: Path to configuration YAML file
- `EXP`: Experiment ID (e.g., THP, NHP, SAHP)
- `DATASET`: Dataset ID (e.g., H2expc, taxi, retweet)
- `PHASE`: Execution phase (train, test, predict, validation, all)
- `CHECKPOINT`: Path to checkpoint file (optional)
- `OUTPUT`: Output directory (optional)
- `DEVICE`: Computation device (auto, cpu, gpu)
- `SEED`: Random seed (optional)
- `VERBOSE`: Enable verbose logging (optional)
- `DIR`: Directory for configuration listing (optional)


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

