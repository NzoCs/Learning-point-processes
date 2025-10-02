# Makefile for EasyTPP
# Automates installation and common tasks with pyproject.toml

.PHONY: help install install-dev install-all test clean setup docs format lint check cli-run cli-interactive cli-list-configs cli-validate cli-info uv-sync uv-lock uv-add uv-remove uv-list uv-pip-list uv-init uv-check uv-clean

# Variables
PYTHON = python
UV = uv
CLI_SCRIPT = scripts/easytpp_cli.py

help: ## Display this help
	@echo "EasyTPP - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-15s %s\n", $$1, $$2}'

install: ## Basic installation (main dependencies only)
	@echo "Installing EasyTPP basic..."
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
	@echo "Complete EasyTPP installation..."
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
	@python -m pytest --cov=easy_tpp --cov-report=html

format: ## Format code with Black
	@echo "Formatting code..."
	@python -m black easy_tpp/ tests/ examples/ scripts/
	@echo "Code formatted!"

format-check: ## Check formatting without modifying
	@echo "Checking formatting..."
	@python -m black --check easy_tpp/ tests/ examples/ scripts/

isort: ## Organize imports
	@echo "Organizing imports..."
	@python -m isort easy_tpp/ tests/ examples/ scripts/
	@echo "Imports organized!"

lint: ## Check code with flake8
	@echo "Checking code with flake8..."
	@python -m flake8 easy_tpp/ tests/ examples/ scripts/
	@echo "Linting completed!"

type-check: ## Check types with mypy
	@echo "Checking types..."
	@python -m mypy easy_tpp

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
	@python -m black easy_tpp/ tests/ examples/ scripts/
	@python -m isort easy_tpp/ tests/ examples/ scripts/
	@echo "Style issues fixed!"

pre-commit: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks..."
	@pre-commit run --all-files

quick-start: ## Quick start guide
	@echo "EasyTPP Quick Start Guide (using uv)"
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
	@echo "      EASYTPP DEMO - ADVANCED TPP FRAMEWORK"
	@echo "=================================================="
	@echo ""
	@echo "INTRODUCTION TO THE FRAMEWORK:"
	@echo "EasyTPP is a modern framework for Temporal Point Processes (TPP)"
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
	@echo "* Interactive CLI for easy usage"
	@echo "* Multi-GPU and distributed training support"
	@echo ""
	@echo "=================================================="
	@echo "STEP 1: Installation Check"
	@echo "=================================================="
	@make check
	@echo ""
	@echo "=================================================="
	@echo "STEP 2: System Information via CLI"
	@echo "=================================================="
	@make cli-info
	@echo ""
	@echo "=================================================="
	@echo "STEP 3: Available Configurations"
	@echo "=================================================="
	@make cli-list-configs
	@echo ""
	@echo "=================================================="
	@echo "STEP 4: Configuration Validation"
	@echo "=================================================="
	@make cli-validate CONFIG=./configs/runner_config.yaml EXP=THP DATASET=H2expc
	@echo ""
	@echo "=================================================="
	@echo "STEP 5: Core Components Test"
	@echo "=================================================="
	@$(PYTHON) -c "import easy_tpp; print('+ EasyTPP successfully imported')"
	@$(PYTHON) -c "from easy_tpp.config_factory import RunnerConfig; print('+ Configuration factory available')"
	@$(PYTHON) -c "from easy_tpp.runner import Runner; print('+ Runner available')"
	@$(PYTHON) -c "from easy_tpp.models.basemodel import Model; print('+ Base models available')"
	@echo ""
	@echo "=================================================="
	@echo "STEP 6: LIVE DEMO - Running THP Model Test"
	@echo "=================================================="
	@echo "Now demonstrating a real TPP experiment..."
	@echo "Running: THP (Transformer Hawkes Process) on test phase"
	@echo ""
	@make cli-run CONFIG=./configs/runner_config.yaml EXP=THP DATASET=test PHASE=test VERBOSE=1
	@echo ""
	@echo "=================================================="
	@echo "NEXT STEPS SUGGESTIONS"
	@echo "=================================================="
	@echo "Now you can:"
	@echo ""
	@echo "Run complete examples:"
	@echo "  make cli-example-thp     # THP (Transformer Hawkes Process)"
	@echo "  make cli-example-nhp     # NHP (Neural Hawkes Process)"
	@echo ""
	@echo "Interactive mode:"
	@echo "  make cli-interactive     # Interactive interface for experiments"
	@echo ""
	@echo "Launch your own experiment:"
	@echo "  make cli-run CONFIG=config.yaml EXP=THP DATASET=H2expc PHASE=train"
	@echo ""
	@echo "Documentation:"
	@echo "  * README.md - Overview and installation"
	@echo "  * SETUP_GUIDE.md - Detailed configuration guide"
	@echo "  * examples/ - Practical examples and notebooks"
	@echo ""
	@echo "CLI Help:"
	@echo "  make cli-help           # Detailed CLI help"
	@echo "  make help              # All Makefile commands"
	@echo ""
	@echo "=================================================="
	@echo "      DEMONSTRATION COMPLETED SUCCESSFULLY!"
	@echo "=================================================="

# ============================================================================
# EASYTPP CLI COMMANDS
# ============================================================================

cli-run: ## [RUN] Execute TPP experiment (usage: make cli-run CONFIG=config.yaml EXP=THP DATASET=H2expc PHASE=test)
	@echo "Launching EasyTPP experiment..."
	@$(PYTHON) $(CLI_SCRIPT) run \
		--config $(or $(CONFIG),./configs/examples_runner_config.yaml) \
		--experiment $(or $(EXP),THP) \
		--dataset $(or $(DATASET),H2expc) \
		--phase $(or $(PHASE),test) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(VERBOSE),--verbose,)

cli-interactive: ## [INT] Interactive mode to configure and launch experiments
	@echo "EasyTPP interactive mode..."
	@$(PYTHON) $(CLI_SCRIPT) interactive

cli-list-configs: ## [LIST] List available configuration files (usage: make cli-list-configs [DIR=./configs])
	@echo "Configuration list..."
	@$(PYTHON) $(CLI_SCRIPT) list-configs $(if $(DIR),--dir $(DIR),--dir ./configs)

cli-validate: ## [OK] Validate configuration file (usage: make cli-validate CONFIG=config.yaml EXP=THP DATASET=H2expc)
	@echo "Configuration validation..."
	@$(PYTHON) $(CLI_SCRIPT) validate \
		--config $(or $(CONFIG),./configs/examples_runner_config.yaml) \
		--experiment $(or $(EXP),THP) \
		--dataset $(or $(DATASET),H2expc) \
		$(if $(VERBOSE),--verbose,)

cli-info: ## [INFO] Display system and environment information
	@echo "System information..."
	@$(PYTHON) $(CLI_SCRIPT) info

cli-help: ## [HELP] Display EasyTPP CLI help
	@echo "EasyTPP CLI help..."
	@$(PYTHON) $(CLI_SCRIPT) --help

# Popular CLI shortcuts
run: cli-run ## [RUN] Shortcut for cli-run
interactive: cli-interactive ## [INT] Shortcut for cli-interactive 
configs: cli-list-configs ## [LIST] Shortcut for cli-list-configs
validate: cli-validate ## [OK] Shortcut for cli-validate
info: cli-info ## [INFO] Shortcut for cli-info

# Predefined examples
cli-example-thp: ## [TEST] Example: THP on H2expc in test mode
	@echo "THP example..."
	@make cli-run CONFIG=./configs/examples_runner_config.yaml EXP=THP DATASET=H2expc PHASE=test

cli-example-nhp: ## [TEST] Example: NHP on H2expc in train mode
	@echo "NHP example..."
	@make cli-run CONFIG=./configs/examples_runner_config.yaml EXP=NHP DATASET=H2expc PHASE=train

cli-quick-test: ## [QUICK] Quick system test with validation
	@echo "Quick system test..."
	@make cli-info
	@echo ""
	@make cli-list-configs
	@echo ""
	@make cli-validate

# ============================================================================

# Targets par défaut
all: install-all setup-dev check test

.DEFAULT_GOAL := help
