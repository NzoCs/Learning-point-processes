# Makefile pour EasyTPP
# Automatise l'installation et les tâches communes avec pyproject.toml

.PHONY: help install install-dev install-all test clean setup docs format lint check

# Variables
PYTHON = python
PIP = pip

# Couleurs pour l'affichage
BLUE = \033[1;34m
GREEN = \033[1;32m
YELLOW = \033[1;33m
RED = \033[1;31m
NC = \033[0m # No Color

help: ## Affiche cette aide
	@echo "$(BLUE)EasyTPP - Commandes disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Installation de base (dépendances principales uniquement)
	@echo "$(BLUE)🔧 Installation de base d'EasyTPP...$(NC)"
	@$(PIP) install -e .
	@echo "$(GREEN)✅ Installation de base terminée!$(NC)"

install-dev: ## Installation avec outils de développement
	@echo "$(BLUE)🛠️ Installation avec outils de développement...$(NC)"
	@$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✅ Installation dev terminée!$(NC)"

install-cli: ## Installation avec outils CLI
	@echo "$(BLUE)📟 Installation avec outils CLI...$(NC)"
	@$(PIP) install -e ".[cli]"
	@echo "$(GREEN)✅ Installation CLI terminée!$(NC)"

install-docs: ## Installation avec outils de documentation
	@echo "$(BLUE)📚 Installation avec outils de documentation...$(NC)"
	@$(PIP) install -e ".[docs]"
	@echo "$(GREEN)✅ Installation docs terminée!$(NC)"

install-all: ## Installation complète (toutes les dépendances)
	@echo "$(BLUE)� Installation complète d'EasyTPP...$(NC)"
	@$(PIP) install -e ".[all]"
	@echo "$(GREEN)✅ Installation complète terminée!$(NC)"

setup-dev: ## Configuration des outils de développement
	@echo "$(BLUE)⚙️ Configuration des outils de développement...$(NC)"
	@pre-commit install
	@echo "$(GREEN)✅ Pre-commit hooks installés!$(NC)"

check: ## Vérification de l'installation
	@echo "$(BLUE)� Vérification de l'installation...$(NC)"
	@$(PYTHON) check_installation.py

test: ## Exécute les tests
	@echo "$(BLUE)🧪 Exécution des tests...$(NC)"
	@pytest

test-cov: ## Exécute les tests avec couverture
	@echo "$(BLUE)📊 Tests avec couverture...$(NC)"
	@pytest --cov=easy_tpp --cov-report=html

format: ## Formate le code avec Black
	@echo "$(BLUE)🎨 Formatage du code...$(NC)"
	@black .
	@echo "$(GREEN)✅ Code formaté!$(NC)"

format-check: ## Vérifie le formatage sans modifier
	@echo "$(BLUE)🔍 Vérification du formatage...$(NC)"
	@black --check .

isort: ## Organise les imports
	@echo "$(BLUE)📋 Organisation des imports...$(NC)"
	@isort .
	@echo "$(GREEN)✅ Imports organisés!$(NC)"

lint: ## Vérifie le code avec flake8
	@echo "$(BLUE)🔍 Vérification du code avec flake8...$(NC)"
	@flake8

type-check: ## Vérifie les types avec mypy
	@echo "$(BLUE)� Vérification des types...$(NC)"
	@mypy easy_tpp

clean: ## Nettoie les fichiers temporaires
	@echo "$(BLUE)🧹 Nettoyage...$(NC)"
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
	@echo "$(GREEN)✅ Nettoyage terminé!$(NC)"

docs: ## Génère la documentation
	@echo "$(BLUE)📚 Génération de la documentation...$(NC)"
	@cd docs && make html
	@echo "$(GREEN)✅ Documentation générée!$(NC)"

docs-serve: ## Serve la documentation localement
	@echo "$(BLUE)🌐 Serveur de documentation...$(NC)"
	@cd docs && sphinx-autobuild . _build/html

check-deps: ## Vérifie les dépendances principales
	@echo "$(BLUE)📦 Vérification des dépendances...$(NC)"
	@$(PYTHON) -c "import torch; print('✅ PyTorch:', torch.__version__)" || echo "❌ PyTorch non installé"
	@$(PYTHON) -c "import pytorch_lightning; print('✅ PyTorch Lightning')" || echo "❌ PyTorch Lightning non installé"
	@$(PYTHON) -c "import numpy; print('✅ NumPy')" || echo "❌ NumPy non installé"
	@$(PYTHON) -c "import pandas; print('✅ Pandas')" || echo "❌ Pandas non installé"
	@$(PYTHON) -c "import easy_tpp; print('✅ EasyTPP')" || echo "❌ EasyTPP non installé"

build: ## Construit le package
	@echo "$(BLUE)� Construction du package...$(NC)"
	@$(PYTHON) -m build
	@echo "$(GREEN)✅ Package construit!$(NC)"

quality: ## Exécute tous les contrôles qualité
	@echo "$(BLUE)� Contrôles qualité...$(NC)"
	@make format-check
	@make lint
	@make type-check
	@make test
	@echo "$(GREEN)✅ Tous les contrôles qualité passés!$(NC)"

pre-commit: ## Exécute les hooks pre-commit sur tous les fichiers
	@echo "$(BLUE)🔄 Exécution des hooks pre-commit...$(NC)"
	@pre-commit run --all-files

quick-start: ## Guide de démarrage rapide
	@echo "$(BLUE)🚀 Guide de Démarrage Rapide EasyTPP$(NC)"
	@echo ""
	@echo "$(YELLOW)Étape 1:$(NC) Installation complète"
	@echo "  make install-all"
	@echo ""
	@echo "$(YELLOW)Étape 2:$(NC) Configuration développement"
	@echo "  make setup-dev"
	@echo ""
	@echo "$(YELLOW)Étape 3:$(NC) Vérification"
	@echo "  make check"
	@echo ""
	@echo "$(YELLOW)Étape 4:$(NC) Tests"
	@echo "  make test"
	@echo ""
	@echo "$(GREEN)📚 Documentation: README.md et SETUP_GUIDE.md$(NC)"

demo: ## Démonstration rapide
	@echo "$(BLUE)🎬 Démonstration EasyTPP$(NC)"
	@echo ""
	@echo "$(YELLOW)1. Vérification de l'installation:$(NC)"
	@make check
	@echo ""
	@echo "$(YELLOW)2. Test d'import:$(NC)"
	@$(PYTHON) -c "import easy_tpp; print('✅ EasyTPP importé avec succès!')"
	@echo ""
	@echo "$(GREEN)✅ Démonstration terminée!$(NC)"

# Targets par défaut
all: install-all setup-dev check test

.DEFAULT_GOAL := help
