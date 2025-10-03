# Makefile for New_LTPP project
# Automates installation and common tasks with pyproject.toml

.PHONY: help install install-dev install-all test clean setup docs format lint check cli-run cli-interactive cli-list-configs cli-validate cli-info uv-sync uv-lock uv-add uv-remove uv-list uv-pip-list uv-init uv-check uv-clean

# Variables
PYTHON = python
UV = uv
CLI_RUNNERS = new_ltpp/scripts/new_ltpp_cli_runners.py

help: ## Display this help
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

install-cli: ## Installation with CLI tools
	@echo "Installing with CLI tools..."
	@$(UV) sync --group cli
	@echo "CLI installation completed!"

install-docs: ## Installation with documentation tools
	@echo "Installing with documentation tools..."
	@$(UV) sync --group docs
	@echo "Docs installation completed!"

install-all: ## Complete installation (all dependencies)
	@echo "Complete new_ltpp installation..."
	@$(UV) sync --all-groups
	@echo "Complete installation finished!"

uv-sync: ## Sync dependencies with uv.lock
	@echo "Syncing dependencies with uv..."
	@$(UV) sync
	@echo "Dependencies synced!"

uv-lock: ## Update uv.lock file
	@echo "Updating uv.lock file..."
	@$(UV) lock
	@echo "Lock file updated!"

uv-add: ## Add new dependency (usage: make uv-add PACKAGE=package_name)
	@echo "Adding package: $(PACKAGE)..."
	@$(UV) add $(PACKAGE)
	@echo "Package added!"

uv-remove: ## Remove dependency (usage: make uv-remove PACKAGE=package_name)
	@echo "Removing package: $(PACKAGE)..."
	@$(UV) remove $(PACKAGE)
	@echo "Package removed!"

uv-list: ## List installed packages with uv
	@echo "Installed packages:"
	@$(UV) tree

uv-pip-list: ## List packages in pip format
	@echo "Packages (pip format):"
	@$(UV) pip list

uv-init: ## Initialize uv project (if needed)
	@echo "Initializing uv project..."
	@$(UV) init
	@echo "Project initialized!"

uv-check: ## Check uv installation and project status
	@echo "Checking uv status..."
	@$(UV) --version
	@echo "Project status:"
	@$(UV) tree --depth 1

uv-clean: ## Clean uv cache
	@echo "Cleaning uv cache..."
	@$(UV) cache clean
	@echo "Cache cleaned!"

setup-dev: ## Configure development tools
	@echo "Configuring development tools..."
	@pre-commit install
	@echo "Pre-commit hooks installed!"

check: ## Verify installation
	@echo "Verifying installation..."
	@$(PYTHON) check_installation.py

test: ## Run tests
	@echo "Running tests..."
	@python -m pytest

test-cov: ## Run tests with coverage
	@echo "Tests with coverage..."
	@python -m pytest --cov=new_ltpp --cov-report=html

format: ## Format code with Black
	@echo "Formatting code..."
	@python -m black new_ltpp/ tests/ examples/ scripts/
	@echo "Code formatted!"

format-check: ## Check formatting without modifying
	@echo "Checking formatting..."
	@python -m black --check new_ltpp/ tests/ examples/ scripts/

isort: ## Organize imports
	@echo "Organizing imports..."
	@python -m isort new_ltpp/ tests/ examples/ scripts/
	@echo "Imports organized!"

lint: ## Check code with flake8
	@echo "Checking code with flake8..."
	@python -m flake8 new_ltpp/ tests/ examples/ scripts/
	@echo "Linting completed!"

type-check: ## Check types with mypy
	@echo "Checking types..."
	@python -m mypy new_ltpp

clean: ## Clean temporary files
	@echo "Cleaning..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.log" -delete
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@echo "Cleaning completed!"

docs: ## Generate documentation
	@echo "Generating documentation..."
	@cd docs && make html
	@echo "Documentation generated!"

docs-serve: ## Serve documentation locally
	@echo "Documentation server..."
	@cd docs && sphinx-autobuild . _build/html

build: ## Build package
	@echo "Building package..."
	@$(PYTHON) -m build
	@echo "Package built!"

quality: ## Run all quality checks
	@echo "Quality checks..."
	@make format-check
	@make lint
	@make type-check
	@make test
	@echo "All quality checks passed!"

fix-style: ## Auto-fix style issues (format + isort)
	@echo "Auto-fixing style issues..."
	@python -m black new_ltpp/ tests/ examples/ scripts/
	@python -m isort new_ltpp/ tests/ examples/ scripts/
	@echo "Style issues fixed!"

pre-commit: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks..."
	@pre-commit run --all-files

quick-start: ## Quick start guide
	@echo "new_ltpp Quick Start Guide (using uv)"
	@echo ""
	@echo "Step 1: Check uv installation"
	@echo "  make uv-check"
	@echo ""
	@echo "Step 2: Sync all dependencies"
	@echo "  make uv-sync"
	@echo ""
	@echo "Step 3: Install with all groups"
	@echo "  make install-all"
	@echo ""
	@echo "Step 4: Development configuration"
	@echo "  make setup-dev"
	@echo ""
	@echo "Step 5: Verification"
	@echo "  make check"
	@echo ""
	@echo "Step 6: Tests"
	@echo "  make test"
	@echo ""
	@echo "Useful uv commands:"
	@echo "  make uv-list     # Show dependency tree"
	@echo "  make uv-add PACKAGE=name  # Add dependency"
	@echo "  make uv-remove PACKAGE=name  # Remove dependency"
	@echo ""
	@echo "Documentation: README.md and SETUP_GUIDE.md"

demo: ## Quick demonstration with framework introduction
	@echo "=================================================="
	@echo "      new_ltpp DEMO - ADVANCED TPP FRAMEWORK"
	@echo "=================================================="
	@echo ""
	@echo "INTRODUCTION TO THE FRAMEWORK:"
	@echo "new_ltpp is a modern framework for Temporal Point Processes (TPP)"
	@echo "that models sequences of events occurring over time."
	@echo ""
	@echo "KEY CONCEPTS:"
	@echo "* TPP: Models when and what type of event will occur"
	@echo "* Supported models: NHP, THP, SAHP, RMTPP, FullyNN, etc."
	@echo "* Lightning Framework: Optimized and scalable training"
	@echo "* Advanced metrics: Wasserstein, MMD, Sinkhorn distance"
	@echo ""
	@echo "ARCHITECTURE:"
	@echo "* YAML configuration to define experiments"
	@echo "* Runner to orchestrate train/test/predict phases"
	@echo "* Modular CLI with dedicated runners"
	@echo "* Multi-GPU and distributed training support"
	@echo ""
	@echo "=================================================="
	@echo "STEP 1: Installation Check"
	@echo "=================================================="
	@make check
	@echo ""
	@echo "=================================================="
	@echo "STEP 2: System Information"
	@echo "=================================================="
	@make info
	@echo ""
	@echo "=================================================="
	@echo "STEP 3: CLI Version"
	@echo "=================================================="
	@make version
	@echo ""
	@echo "=================================================="
	@echo "STEP 4: Core Components Test"
	@echo "=================================================="
	@$(PYTHON) -c "import new_ltpp; print('+ new_ltpp successfully imported')"
	@$(PYTHON) -c "from new_ltpp.configs import ConfigFactory; print('+ Configuration factory available')"
	@$(PYTHON) -c "from new_ltpp.runners import RunnerManager; print('+ Runner manager available')"
	@$(PYTHON) -c "from new_ltpp.model.basemodel import Model; print('+ Base models available')"
	@echo ""
	@echo "=================================================="
	@echo "STEP 5: LIVE DEMO - Quick Experiment"
	@echo "=================================================="
	@echo "Running: NHP (Neural Hawkes Process) - Quick test"
	@echo ""
	@make run-quick
	@echo ""
	@echo "=================================================="
	@echo "NEXT STEPS SUGGESTIONS"
	@echo "=================================================="
	@echo "Now you can:"
	@echo ""
	@echo "Run complete examples:"
	@echo "  make run-nhp          # NHP (Neural Hawkes Process)"
	@echo "  make run-thp          # THP (Transformer Hawkes Process)"
	@echo "  make run-rmtpp        # RMTPP (Recurrent Marked TPP)"
	@echo ""
	@echo "Run benchmarks:"
	@echo "  make benchmark-all    # All benchmarks"
	@echo "  make benchmark-multi  # Multi-config benchmarks"
	@echo ""
	@echo "Interactive mode:"
	@echo "  make setup            # Interactive setup wizard"
	@echo ""
	@echo "Generate data:"
	@echo "  make generate         # Generate synthetic data"
	@echo ""
	@echo "Documentation:"
	@echo "  * README.md - Overview and installation"
	@echo "  * SETUP.md - Detailed configuration guide"
	@echo "  * examples/ - Practical examples and notebooks"
	@echo ""
	@echo "CLI Help:"
	@echo "  make cli-help         # Detailed CLI help"
	@echo "  make help             # All Makefile commands"
	@echo ""
	@echo "=================================================="
	@echo "      DEMONSTRATION COMPLETED SUCCESSFULLY!"
	@echo "=================================================="

# ============================================================================
# new_ltpp CLI COMMANDS (New Runner Architecture)
# ============================================================================

# ---------- EXPERIMENT RUNNER ----------

run: ## [RUN] Run TPP experiment (usage: make run DATA=test MODEL=neural_small EPOCHS=10)
	@echo ">> Lancement d'une experience TPP..."
	@$(PYTHON) $(CLI_RUNNERS) run \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(DATA),--data-config $(DATA),--data-config test) \
		$(if $(MODEL),--model-config $(MODEL),--model-config neural_small) \
		$(if $(TRAINING),--training-config $(TRAINING),--training-config quick_test) \
		$(if $(DATA_LOADING),--data-loading-config $(DATA_LOADING),--data-loading-config quick_test) \
		$(if $(SIMULATION),--simulation-config $(SIMULATION),) \
		$(if $(THINNING),--thinning-config $(THINNING),) \
		$(if $(LOGGER),--logger-config $(LOGGER),--logger-config tensorboard) \
		$(if $(MODEL_ID),--model $(MODEL_ID),--model NHP) \
		$(if $(PHASE),--phase $(PHASE),--phase all) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(SAVE_DIR),--save-dir $(SAVE_DIR),) \
		$(if $(GPU),--gpu $(GPU),) \
		$(if $(DEBUG),--debug,)

run-debug: ## [DEBUG] Run experiment with debug mode
	@make run DEBUG=1

run-train: ## [TRAIN] Run only training phase
	@make run PHASE=train

run-test: ## [TEST] Run only test phase
	@make run PHASE=test

run-predict: ## [PREDICT] Run only prediction phase
	@make run PHASE=predict

# Quick examples

run-nhp: ## [NHP] Run NHP model on test dataset
	@make run MODEL_ID=NHP DATA=test MODEL=neural_small EPOCHS=10

run-rmtpp: ## [RMTPP] Run RMTPP model on test dataset
	@make run MODEL_ID=RMTPP DATA=test MODEL=neural_small EPOCHS=10

run-thp: ## [THP] Run THP model on test dataset
	@make run MODEL_ID=THP DATA=test MODEL=neural_large EPOCHS=20

# ---------- BENCHMARK RUNNER ----------

benchmark: ## [BENCH] Run benchmarks (usage: make benchmark DATA=test)
	@echo ">> Lancement des benchmarks TPP..."
	@$(PYTHON) $(CLI_RUNNERS) benchmark \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(DATA),--data-config $(DATA),--data-config test) \
		$(if $(DATA_LOADING),--data-loading-config $(DATA_LOADING),--data-loading-config quick_test) \
		$(if $(BENCHMARKS),--benchmarks $(BENCHMARKS),) \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(ALL),--all,) \
		$(if $(ALL_CONFIGS),--all-configs,) \
		$(if $(DEBUG),--debug,)

benchmark-all: ## [BENCH-ALL] Run all benchmarks on one config
	@make benchmark ALL=1

benchmark-list: ## [BENCH-LIST] List available benchmarks
	@$(PYTHON) $(CLI_RUNNERS) benchmark --list

benchmark-mean: ## [BENCH-MEAN] Run mean inter-time benchmark
	@make benchmark BENCHMARKS=mean_inter_time

benchmark-multi: ## [BENCH-MULTI] Run benchmarks on multiple configs
	@echo ">> Benchmarks multi-configurations..."
	@$(PYTHON) $(CLI_RUNNERS) benchmark \
		--data-config test --data-config large \
		--all --all-configs

# ---------- DATA INSPECTOR ----------

inspect: ## [INSPECT] Inspect TPP data (usage: make inspect DIR=./data/test)
	@echo ">> Inspection des donnees TPP..."
	@$(PYTHON) $(CLI_RUNNERS) inspect $(DIR) \
		$(if $(FORMAT),--format $(FORMAT),--format json) \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(NOSAVE),--no-save,--save) \
		$(if $(SHOW),--show,--no-show) \
		$(if $(MAX_SEQ),--max-seq $(MAX_SEQ),) \
		$(if $(DEBUG),--debug,)

# ---------- DATA GENERATOR ----------

generate: ## [GEN] Generate synthetic TPP data (usage: make generate NUM=1000)
	@echo ">> Generation de donnees synthetiques..."
	@$(PYTHON) $(CLI_RUNNERS) generate \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(NUM),--num-seq $(NUM),--num-seq 1000) \
		$(if $(MAX_LEN),--max-len $(MAX_LEN),--max-len 100) \
		$(if $(TYPES),--event-types $(TYPES),--event-types 5) \
		$(if $(METHOD),--method $(METHOD),--method nhp) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(DEBUG),--debug,)

generate-small: ## [GEN-S] Generate small synthetic dataset (100 sequences)
	@make generate NUM=100 MAX_LEN=50

generate-large: ## [GEN-L] Generate large synthetic dataset (10000 sequences)
	@make generate NUM=10000 MAX_LEN=200

# ---------- SYSTEM INFO ----------

info: ## [INFO] Display system and environment information
	@echo ">> Informations systeme..."
	@$(PYTHON) $(CLI_RUNNERS) info \
		$(if $(NODEPS),--no-deps,--deps) \
		$(if $(NOHW),--no-hw,--hw) \
		$(if $(OUTPUT),--output $(OUTPUT),)

# ---------- INTERACTIVE SETUP ----------

setup: ## [SETUP] Interactive setup wizard
	@echo ">> Configuration interactive..."
	@$(PYTHON) $(CLI_RUNNERS) setup \
		$(if $(TYPE),--type $(TYPE),--type experiment) \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(QUICK),--quick,)

setup-quick: ## [SETUP-Q] Quick setup mode
	@make setup QUICK=1

# ---------- VERSION ----------

version: ## [VERSION] Display CLI version
	@$(PYTHON) $(CLI_RUNNERS) version

cli-help: ## [HELP] Display new_ltpp CLI help
	@echo ">> Aide du CLI new_ltpp..."
	@$(PYTHON) $(CLI_RUNNERS) --help

# ---------- COMBINED WORKFLOWS ----------

full-pipeline: ## [PIPELINE] Complete pipeline: train -> test -> predict
	@echo ">> Pipeline complet..."
	@make run-train
	@make run-test
	@make run-predict

experiment-with-benchmark: ## [EXP+BENCH] Run experiment then benchmarks
	@echo ">> Experience + Benchmarks..."
	@make run-quick
	@make benchmark-all

quick-demo: ## [DEMO] Quick demonstration of all features
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
	@make run-quick
	@echo ""
	@echo "4. Benchmarks"
	@make benchmark-all
	@echo ""
	@echo "=================================================="
	@echo "      DEMO COMPLETED!"
	@echo "=================================================="

# Popular shortcuts
cli-run: run ## Shortcut for run
cli-benchmark: benchmark ## Shortcut for benchmark
cli-inspect: inspect ## Shortcut for inspect
cli-generate: generate ## Shortcut for generate
cli-info: info ## Shortcut for info
cli-setup: setup ## Shortcut for setup

# ============================================================================

# Default targets
all: install-all setup-dev check test

.DEFAULT_GOAL := help