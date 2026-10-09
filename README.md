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

**Prerequisites:** Python **3.11** and `uv`.

### 1. Using `uv` (Recommended)

This project uses `uv` for dependency management.

```bash
# Download into a fresh directory, then install the committed lock
git clone --branch codex/pysiglib-migration https://github.com/NzoCs/Learning-point-processes.git
cd Learning-point-processes
uv sync --frozen --python 3.11 --no-default-groups --no-build-package pysiglib
uv run --frozen --no-sync new-ltpp --help
```

### 2. Development and native dependencies

```bash
uv sync --frozen --group dev
uv run --frozen --no-sync python -m pytest tests -m "not slow"
```

This branch uses **pySigLib 4.0.0** for signature kernels, with the historical
finite-difference solver and unbiased MMD² estimator. The Ruche GPU profile
requires the locked CUDA plugin: `uv sync --frozen --no-default-groups --extra ruche
--no-build-package pysiglib --no-build-package pysiglib-cuda`.
There is no automatic CPU or legacy fallback. pySigLib is the only signature
backend supported by this branch. The historical implementation is preserved on
`codex/reproductibilite-ruche`. Read the [migration validation](docs/PYSIGLIB_MIGRATION.md)
and the [Ruche storage and installation procedure](docs/RUCHE.md).

---

## Reproducible runs

Training accepts `--seed` and an explicit `--checkpoint`. Each attempt writes to
its own run directory, containing `effective_config.yaml`, `manifest.json`,
checkpoints and results. The manifest records versions, configuration, dataset
revision, local data hashes and the checkpoint used by each completed phase.
The saved effective configuration can be passed directly to `run --config`:

```bash
uv run --frozen --no-sync new-ltpp run --config /path/to/effective_config.yaml \
  --seed 42 --phase all --save-dir artifacts/replayed
# Resume at an epoch boundary from the last training checkpoint:
uv run --frozen --no-sync new-ltpp run --config /path/to/effective_config.yaml \
  --checkpoint /path/to/last.ckpt --epochs 10 --phase train --save-dir artifacts/resumed
```

The small complete CPU pipeline is tested offline with local fixture data:

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg uv run --frozen --no-sync python -m pytest \
  -o addopts= tests/test_reproducibility.py -q
```

`deterministic: true` in the training YAML enables strict PyTorch determinism.
CPU repeatability and epoch-boundary resume are verified on the small NHP case
with zero loader workers; CUDA, DDP and multi-worker resume still need validation.

An opt-in integration regression suite covers every public model using fixed,
small offline data and versioned numerical references. It is separate from the
regular tests and runs manually through GitHub Actions or locally:

```bash
uv run --frozen --no-sync python -m scripts.integration_suite
```

See [integration tests](docs/INTEGRATION_TESTS.md) for the checked outputs,
tolerances, artefacts and the explicit reference-update policy.
The current global coverage is 62.86%, below the retained 80% gate, so the normal
coverage command is expected to fail until coverage improves. See the
[reproducibility report](rapports/REPRODUCTIBILITE_RUCHE.md) for remaining limits
and the numerical changes to MKernel and p-values.

## ⚡ Quick Start

### 1. Run a Demo (NHP on Test Data)

To verify everything is working, use the Makefile target that runs a quick end-to-end pipeline (Train → Test → Predict):

```bash
make run-demo
```

Run this after installation with `uv run --frozen --no-sync make run-demo`.
The `test` dataset is downloaded from Hugging Face (`NzoCs/test_dataset`).
This demo requires network access or a populated dataset cache.
Outputs are written beneath `artifacts/test/NHP_.../` in the current working directory.

### 2. Run an Experiment via CLI

You can run experiments directly using the `new-ltpp` command (or `scripts/cli.py`).

## Example: Train THP on the Taxi dataset

```bash
# Using the installed script entry point
new-ltpp run --model THP --dataset-id taxi --phase train --epochs 50

# OR using the python script directly
uv run --frozen --no-sync python -m scripts.cli run --model THP --dataset-id taxi --phase train --epochs 50
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
| **`generate`** | Generate synthetic TPP data (Hawkes, etc.). | `new-ltpp generate --model hawkes --num-sim 1000` |
| **`benchmark`** | Run naïve benchmarks. | `new-ltpp benchmark --dataset-id test` |
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

---

## ⚙️ Configuration

We use a hierarchical YAML configuration system located in `yaml_configs/configs.yaml`.

You can override configurations via CLI arguments (e.g., `--training-config quick_test`) or by creating your own YAML files.

**Key Config Sections:**

* **Data Config**: Dataset paths and formats (`test`, `taxi`, `retweet`).
* **Model Config**: Hyperparameters for each model (`NHP`, `THP`).
* **Training Config**: Epochs and learning rate (`quick_test`, `e500_b1`).
* **Data Loading Config**: Batch size and workers (`quick_test`, `b32_w1`).
* **Simulation Config**: RNG seed (`fixed_events`, `quick_test`, `debug`).
* **Signature representation**: `embedding_type: counting_grid`.

For an offline CPU example using the current configuration API, open
[the getting-started notebook](notebooks/NewLTPP_Getting_Started.ipynb).
The [configuration contract](docs/CONFIGURATION_CONTRACTS.md) describes how to
adapt a copy of an old configuration while preserving its archived provenance.

---

## 📁 Artifacts & Logging

All results are saved in the `artifacts/` directory by default (configurable via `--save-dir`).

* **Checkpoints**: Best model weights.
* **Logs**: TensorBoard logs (view with `tensorboard --logdir artifacts`).
* **Results**: JSON files with metrics and prediction outputs.

---

## 📄 License

MIT License
