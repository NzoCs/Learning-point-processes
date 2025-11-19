# Makefile for New_LTPP project
# Automates installation and common tasks with pyproject.toml

.PHONY: help install install-dev lint format type-check run run-nhp run-thp run-rmtpp

# Variables
UV = uv
CLI = new-ltpp

help: ## Display available simplified commands
	@echo "new_ltpp - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-15s %s\n", $$1, $$2}'

install: ## Basic installation (main dependencies only)
	@echo "Installing new_ltpp basic..."
	@$(UV) sync
	@echo "Basic installation completed!"

install-dev: ## Installation with development tools
	@echo "Installing with development tools..."
	@$(UV) sync --group dev
	@echo "Dev installation completed!"

lint: ## Run linting (flake8)
	@echo "Running flake8..."
	@python -m flake8 new_ltpp/ tests/ examples/ || true
	@echo "Linting finished (errors reported above)."

format: ## Format code (black + isort)
	@echo "Formatting code with black and isort..."
	@python -m black new_ltpp/ tests/ examples/
	@python -m isort new_ltpp/ tests/ examples/
	@echo "Formatting completed."

type-check: ## Type check with mypy
	@echo "Running mypy type checks..."
	@python -m mypy new_ltpp/ tests/ examples/ || true
	@echo "Type checking finished (issues reported above)."

# Generic run target (defaults to demo/test configs)
run: ## Run full pipeline via CLI (pass variables to override)
	@$(CLI) run \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(DATA),--data-config $(DATA),--data-config test) \
		$(if $(MODEL),--model-config $(MODEL),--model-config quick_test) \
		$(if $(TRAINING),--training-config $(TRAINING),--training-config quick_test) \
		$(if $(DATA_LOADING),--data-loading-config $(DATA_LOADING),--data-loading-config quick_test) \
		$(if $(SIMULATION),--simulation-config $(SIMULATION),--simulation-config quick_test) \
		$(if $(THINNING),--thinning-config $(THINNING),--thinning-config quick_test) \
		$(if $(LOGGER),--logger-config $(LOGGER),--logger-config tensorboard) \
		$(if $(MODEL_ID),--model $(MODEL_ID),--model NHP) \
		--phase all \
		$(if $(EPOCHS),--epochs $(EPOCHS),--epochs 5) \
		$(if $(SAVE_DIR),--save-dir $(SAVE_DIR),) \
		$(if $(GPU),--gpu $(GPU),) \
		$(if $(DEBUG),--debug,)

# Model-specific shortcuts that run the entire pipeline with sensible defaults
run-nhp: ## Run full pipeline for NHP model (demo uses `test` configs)
	@$(MAKE) run MODEL_ID=NHP MODEL=neural_small DATA=test TRAINING=quick_test DATA_LOADING=quick_test SIMULATION=quick_test THINNING=quick_test LOGGER=tensorboard

run-thp: ## Run full pipeline for THP model (demo uses `test` configs)
	@$(MAKE) run MODEL_ID=THP MODEL=neural_large DATA=test TRAINING=quick_test DATA_LOADING=quick_test SIMULATION=quick_test THINNING=quick_test LOGGER=tensorboard

run-rmtpp: ## Run full pipeline for RMTPP model (demo uses `test` configs)
	@$(MAKE) run MODEL_ID=RMTPP MODEL=neural_small DATA=test TRAINING=quick_test DATA_LOADING=quick_test SIMULATION=quick_test THINNING=quick_test LOGGER=tensorboard


benchmark-list: ## [BENCH-LIST] List available benchmarks
	@$(CLI) benchmark --list

benchmark-mean: ## [BENCH-MEAN] Run mean inter-time benchmark
	@make benchmark BENCHMARKS=mean_inter_time

benchmark-multi: ## [BENCH-MULTI] Run benchmarks on multiple configs
	@echo ">> Benchmarks multi-configurations..."
	@$(CLI) benchmark \
		--data-config test --data-config large \
		--all --all-configs

# ---------- SYSTEM INFO ----------

info: ## [INFO] Display system and environment information
	@echo ">> Informations systeme..."
	@$(CLI) info \
		$(if $(NODEPS),--no-deps,--deps) \
		$(if $(NOHW),--no-hw,--hw) \
		$(if $(OUTPUT),--output $(OUTPUT),)

# ---------- INTERACTIVE SETUP ----------

setup: ## [SETUP] Interactive setup wizard
	@echo ">> Configuration interactive..."
	@$(CLI) setup \
		$(if $(TYPE),--type $(TYPE),--type experiment) \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(QUICK),--quick,)

setup-quick: ## [SETUP-Q] Quick setup mode
	@make setup QUICK=1

# ---------- VERSION ----------

version: ## [VERSION] Display CLI version
	@$(CLI) version

cli-help: ## [HELP] Display new_ltpp CLI help
	@echo ">> Aide du CLI new_ltpp..."
	@$(CLI) --help

demo: ## [DEMO] Quick demonstration of all features
	@echo "=================================================="
	@echo "      new_ltpp DEMO - NEW CLI ARCHITECTURE"
	@echo "=================================================="
	@echo ""
	@echo "1. System Information"
	@make info
	@echo ""
	@echo "2. CLI Version"
	@make version
	@echo ""
	@echo "3. Quick Training (5 epochs)"
	@make run-nhp EPOCHS=5
	@echo ""
	@echo "4. Benchmarks"
	@make benchmark-list
	@echo ""
	@echo "=================================================="
	@echo "      DEMO COMPLETED!"
	@echo "=================================================="

# ============================================================================

.DEFAULT_GOAL := help