# New-LTPP: Advanced Temporal Point Process Framework

**New-LTPP** is a modern and flexible framework for learning, simulating, and analyzing **Temporal Point Processes (TPP)**. It is built on **PyTorch** and **PyTorch Lightning** for scalability and research efficiency.

## 🚀 Core Capabilities

New-LTPP is designed to be a comprehensive toolkit for TPP research, covering the entire lifecycle from training to advanced evaluation.

### 1. Training 🏋️

* **Multi-Model Support**: Train various Neural TPP models (NHP, THP, ODETPP, etc.) on standard or custom datasets.
* **PyTorch Lightning**: Benefits from distributed training, automatic checkpointing, and robust logging.
* **Flexible Configs**: Easy hyperparameter tuning via YAML files or CLI overrides.

### 2. Simulation 🎲

* **Synthetic Data Generation**: Generate event sequences using known processes (Hawkes, Self-Correcting) for controlled experiments.
* **Path Simulation**: Simulate entire future trajectories (event sequences) from trained models to analyze long-term dynamics, not just next-step predictions.

### 3. Evaluation & Analysis 📊

* **Distribution Matching**: Evaluate models by comparing the distributions of simulated sequences against real data (e.g., inter-event times, event types).
* **Prediction Metrics**: Standard metrics for next-event prediction (RMSE, Accuracy).
* **🚧 Goodness of Fit (WIP)**: We are currently developing statistical tests (e.g., KS test, QQ plots) to rigorously quantify model fit.

---

## 🛠️ Installation

**Prerequisites:** Python 3.11+

### 1. Using `uv` (Recommended)

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

### 2. Using `pip`

```bash
pip install -e .
```

---

## ⚡ Quick Start

### 1. Run a Demo (NHP on Test Data)

To verify everything is working, use the Makefile target that runs a quick end-to-end pipeline (Train → Test → Predict):

```bash
make run-demo
```

*Creates artifacts in `artifacts/test/NHP_.../`*

### 2. Run an Experiment via CLI

You can run experiments directly using the `new-ltpp` command (or `scripts/cli.py`).

## Example: Train THP on the Taxi dataset

```bash
# Using the installed script entry point
new-ltpp run --model THP --data-config taxi --phase train --epochs 50

# OR using the python script directly
python scripts/cli.py run --model THP --data-config taxi --phase train --epochs 50
```

### 3. Interactive Setup

If you are unsure about parameters, use the interactive wizard:

```bash
new-ltpp setup
# Follow the prompts to configure your experiment
```

---

## 💻 CLI Commands

The framework provides a unified CLI `new-ltpp` (or `python scripts/cli.py`).

| Command | Description | Example |
| :--- | :--- | :--- |
| **`run`** | Run a TPP experiment (train/test/predict). | `new-ltpp run --model NHP --phase all` |
| **`inspect`** | Inspect and visualize dataset statistics. | `new-ltpp inspect data/taxi --save` |
| **`generate`** | Generate synthetic TPP data (Hawkes, etc.). | `new-ltpp generate --method hawkes --num-sim 1000` |
| **`benchmark`** | Run performance benchmarks. | `new-ltpp benchmark --data-config test` |
| **`setup`** | Launch interactive configuration wizard. | `new-ltpp setup` |
| **`info`** | Display system and environment info. | `new-ltpp info` |

---

## 📚 Model List

Implemented models in `new_ltpp/models/`:

| Model | Paper | Implementation |
| :--- | :--- | :--- |
| **RMTPP** | [KDD'16](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) | `rmtpp.py` |
| **NHP** | [NeurIPS'17](https://arxiv.org/abs/1612.09328) | `nhp.py` |
| **FullyNN** | [NeurIPS'19](https://arxiv.org/abs/1905.09690) | `fullynn.py` |
| **SAHP** | [ICML'20](https://arxiv.org/abs/1907.07561) | `sahp.py` |
| **THP** | [ICML'20](https://arxiv.org/abs/2002.09291) | `thp.py` |
| **IntFree** | [ICLR'20](https://arxiv.org/abs/1909.12127) | `intensity_free.py` |
| **ODETPP** | [ICLR'21](https://arxiv.org/abs/2011.04583) | `ode_tpp.py` |
| **AttNHP** | [ICLR'22](https://arxiv.org/abs/2201.00044) | `attnhp.py` |
| **Hawkes** | Classical | `hawkes.py` |

---

## ⚙️ Configuration

We use a hierarchical YAML configuration system located in `yaml_configs/configs.yaml`.

You can override configurations via CLI arguments (e.g., `--training-config quick_test`) or by creating your own YAML files.

**Key Config Sections:**

* **Data Config**: Dataset paths and formats (`test`, `taxi`, `retweet`).
* **Model Config**: Hyperparameters for each model (`NHP`, `THP`).
* **Training Config**: Epochs, batch size, learning rate (`quick_test`, `full_training`).

---

## 📁 Artifacts & Logging

All results are saved in the `artifacts/` directory by default (configurable via `--save-dir`).

* **Checkpoints**: Best model weights.
* **Logs**: TensorBoard logs (view with `tensorboard --logdir artifacts`).
* **Results**: JSON files with metrics and prediction outputs.

---

## 🤝 Contributing

Feel free to open issues or submit PRs for new models or features.

## 📄 License

MIT License
