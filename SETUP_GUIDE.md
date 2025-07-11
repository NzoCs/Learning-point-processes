# EasyTPP - Guide de Configuration Rapide

Ce guide vous aide à configurer rapidement le projet EasyTPP avec le nouveau système `pyproject.toml`.

## Prérequis

- Python 3.8 ou supérieur
- pip 21.3+ (pour le support complet de pyproject.toml)
- Git

## Installation Rapide

### 1. Cloner le projet

```bash
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Windows :
venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate
```

### 3. Installer le projet

```bash
# Installation complète (recommandée pour le développement)
pip install -e ".[all]"

# Ou installation minimale
pip install -e .
```

### 4. Vérifier l'installation

```bash
python check_installation.py
```

## Options d'Installation

Choisissez l'installation qui correspond à vos besoins :

```bash
# Installation de base (dépendances principales uniquement)
pip install -e .

# Outils de développement (tests, linting, formatage)
pip install -e ".[dev]"

# Outils CLI (interfaces en ligne de commande)
pip install -e ".[cli]"

# Outils de documentation
pip install -e ".[docs]"

# Tout installer
pip install -e ".[all]"
```

## Configuration des Outils de Développement

Le projet inclut des outils préconfigurés pour le développement :

### Makefile (Unix/Linux/Windows avec Git Bash)

Le projet utilise un Makefile pour automatiser les tâches courantes. Sur Windows, vous devez avoir installé :
- `make` (installé via `winget install ezwinports.make`)
- Les outils Unix de Git Bash (inclus avec Git)

**Configuration automatique sur Windows :**
```bash
# Ajouter make et les outils Unix au PATH
$makePath = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe\bin"
$gitUnixPath = "C:\Program Files\Git\usr\bin"
$env:PATH += ";$makePath;$gitUnixPath"
```

**Commandes Makefile disponibles :**
```bash
make help          # Afficher l'aide
make install-all   # Installation complète
make check         # Vérification de l'installation
make test          # Exécuter les tests
make format        # Formater le code
make lint          # Vérifier le code
make clean         # Nettoyer les fichiers temporaires
make demo          # Démonstration rapide
```

### Pre-commit hooks

```bash
# Installer les hooks pre-commit (après avoir installé les dépendances dev)
pre-commit install
```

### Outils disponibles

- **black** : Formatage automatique du code
- **isort** : Organisation des imports
- **flake8** : Vérification de la qualité du code
- **mypy** : Vérification des types statiques
- **pytest** : Tests avec couverture

### Utilisation des outils

```bash
# Formater le code
black .

# Organiser les imports
isort .

# Vérifier le code
flake8

# Vérifier les types
mypy easy_tpp

# Lancer les tests
pytest
```

## Structure du Projet

```
EasyTemporalPointProcess/
├── pyproject.toml          # Configuration principale du projet
├── check_installation.py   # Script de vérification
├── README.md              # Documentation principale
├── easy_tpp/              # Code source principal
├── examples/              # Exemples d'utilisation
├── tests/                 # Tests unitaires
└── docs/                  # Documentation
```

## Configuration pyproject.toml

Toute la configuration du projet est centralisée dans `pyproject.toml` :

- Configuration du système de build
- Dépendances et groupes de dépendances optionnelles
- Configuration des outils (black, isort, pytest, mypy, etc.)
- Métadonnées du projet et URLs

## Groupes de Dépendances

- **`cli`** : Interfaces terminal riches, outils en ligne de commande
- **`docs`** : Système de documentation Sphinx, thèmes et extensions
- **`dev`** : Outils de workflow de développement (tests, linting, formatage)
- **`all`** : Installe toutes les dépendances optionnelles

## Résolution de Problèmes

### Erreur de version Python

```bash
# Vérifier votre version Python
python --version

# Mise à jour recommandée vers Python 3.8+
```

### Erreur de pip

```bash
# Mettre à jour pip
python -m pip install --upgrade pip

# Utiliser python -m pip au lieu de pip directement
python -m pip install -e ".[all]"
```

### Problèmes d'environnement virtuel

```bash
# Recréer l'environnement virtuel
rm -rf venv  # ou rmdir /s venv sur Windows
python -m venv venv
# Réactiver et réinstaller
```

## Support

Si vous rencontrez des problèmes :

1. Vérifiez que Python 3.8+ est installé
2. Lancez `python check_installation.py` pour diagnostiquer
3. Consultez la documentation complète dans README.md
4. Créez une issue sur GitHub si le problème persiste

---
