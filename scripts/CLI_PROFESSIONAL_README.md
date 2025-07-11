# 🚀 EasyTPP CLI - Professional Command Line Interface

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CLI Version](https://img.shields.io/badge/CLI%20Version-2.0-brightgreen.svg)](#)

Un outil en ligne de commande professionnel et moderne pour exécuter des expériences de Processus Ponctuels Temporels (TPP) avec EasyTPP.

## ✨ Fonctionnalités Principales

- 🎯 **Interface CLI Professionnelle** : Commandes intuitives avec aide complète
- 🎨 **Sortie Terminal Enrichie** : Tableaux colorés, barres de progression et affichage stylé
- 🔧 **Mode Interactif** : Configuration guidée pas à pas
- 📁 **Gestion de Configuration** : Système de templates et validation
- 🚀 **Modes d'Exécution Multiples** : Train, test, predict, validate et plus
- 🛡️ **Gestion d'Erreurs Robuste** : Rapports d'erreurs détaillés et récupération
- 📊 **Logging Professionnel** : Rotation et rétention des logs
- 🔄 **Multi-Plateforme** : Compatible Windows, macOS et Linux

## 📦 Installation Rapide

1. **Installation des dépendances** :

   ```bash
   python setup_cli.py
   ```

2. **Vérification de l'installation** :

   ```bash
   python easytpp_cli.py --version
   ```

3. **Test rapide** :

   ```bash
   python easytpp_cli.py --help
   ```

## 🎯 Démarrage Rapide

### Mode Interactif (Recommandé pour les débutants)

```bash
python easytpp_cli.py interactive
```

Le mode interactif vous guide à travers :

- Sélection du fichier de configuration
- Choix de l'expérience et du dataset
- Configuration des paramètres
- Confirmation d'exécution

### Utilisation de Base

```bash
# Exécuter une expérience de test
python easytpp_cli.py run --config configs/runner_config.yaml --experiment THP --dataset H2expc --phase test

# Lister les configurations disponibles
python easytpp_cli.py list-configs --dir configs

# Valider une configuration
python easytpp_cli.py validate --config configs/runner_config.yaml --experiment THP --dataset H2expc
```

### Utilisation Avancée

```bash
# Entraînement avec checkpoint et répertoire de sortie personnalisé
python easytpp_cli.py run \\
  --config configs/advanced_config.yaml \\
  --experiment TransformerHP \\
  --dataset taxi \\
  --phase train \\
  --checkpoint checkpoints/model_epoch_50.ckpt \\
  --output experiments/run_001 \\
  --device gpu \\
  --seed 42

# Exécuter toutes les phases séquentiellement
python easytpp_cli.py run \\
  --config configs/full_pipeline.yaml \\
  --experiment THP \\
  --dataset H2expc \\
  --phase all
```

## 📋 Commandes Disponibles

### `run` - Exécuter des Expériences TPP

Exécute des expériences de processus ponctuels temporels avec des options de configuration complètes.

```bash
python easytpp_cli.py run [OPTIONS]
```

**Options** :

- `--config, -c` : Chemin vers le fichier YAML de configuration (requis)
- `--experiment, -e` : ID de l'expérience dans le fichier config (requis)
- `--dataset, -d` : ID du dataset dans le fichier config (requis)
- `--phase, -p` : Phase à exécuter [train|test|predict|validation|all] (défaut: test)
- `--checkpoint` : Chemin vers le fichier checkpoint
- `--output` : Répertoire de sortie pour les résultats
- `--device` : Dispositif à utiliser [cpu|gpu|auto] (défaut: auto)
- `--seed` : Graine aléatoire pour la reproductibilité

### `interactive` - Mode de Configuration Interactif

Lance un mode interactif pour la configuration guidée des expériences.

```bash
python easytpp_cli.py interactive
```

### `list-configs` - Lister les Configurations Disponibles

Affiche tous les fichiers de configuration disponibles dans un répertoire.

```bash
python easytpp_cli.py list-configs [--dir DIRECTORY]
```

### `validate` - Valider une Configuration

Valide un fichier de configuration et affiche un résumé.

```bash
python easytpp_cli.py validate --config CONFIG --experiment EXP --dataset DATA
```

### `info` - Informations Système

Affiche les informations système et d'environnement incluant PyTorch, CUDA et matériel.

```bash
python easytpp_cli.py info
```

## ⚙️ Configuration

### Structure du Fichier de Configuration

Les fichiers de configuration YAML suivent cette structure :

```yaml
base_config:
  seed: 42
  device: auto
  log_level: INFO

experiments:
  THP:
    model_config:
      model_name: THP
      hidden_size: 64
      num_layers: 4
      # ... plus de paramètres du modèle
      
    data_config:
      dataset_name: H2expc
      batch_size: 32
      # ... plus de paramètres de données
      
    training_config:
      optimizer: Adam
      learning_rate: 0.001
      # ... plus de paramètres d'entraînement

datasets:
  H2expc:
    path: ./data/h2expc
    type: point_process
    # ... configuration du dataset
```

### Configuration CLI

Les paramètres globaux du CLI peuvent être configurés dans `configs/cli_config.yaml` :

```yaml
cli:
  default_config_dir: ./configs
  default_output_dir: ./outputs
  use_rich: true
  color_theme: default

defaults:
  experiment_id: THP
  dataset_id: H2expc
  phase: test
```

## 🎨 Fonctionnalités du Terminal Enrichi

Quand le package `rich` est installé, le CLI fournit :

- **Sortie Colorée** : Coloration syntaxique et texte coloré
- **Barres de Progression** : Suivi en temps réel des progrès
- **Tableaux** : Affichage formaté des données
- **Panneaux** : Affichage organisé des informations
- **Prompts Interactifs** : Saisie utilisateur améliorée

## 🛠️ Scripts d'Assistance

### Windows

- **PowerShell** : `easytpp.ps1` - Script PowerShell professionnel
- **Batch** : `easytpp.bat` - Script batch simple
- **Moderne** : `easytpp_modern_cli.py` - CLI alternatif avec Typer

### Unix/Linux/macOS

- **Shell** : `easytpp` - Script shell exécutable
- **Python** : `easytpp_cli.py` - Script principal

## 📁 Structure des Répertoires

```
New_LTPP/
├── easytpp_cli.py              # CLI principal
├── easytpp_modern_cli.py       # CLI alternatif (Typer)
├── setup_cli.py               # Script d'installation
├── easytpp.ps1                # Wrapper PowerShell
├── easytpp.bat                # Wrapper Batch
├── requirements-cli.txt        # Dépendances CLI
├── CLI_README.md              # Documentation détaillée
├── configs/
│   ├── runner_config_template.yaml
│   └── cli_config.yaml
├── examples/
│   ├── basic_example.py
│   └── advanced_pipeline.py
├── outputs/                   # Résultats d'expériences
├── logs/                      # Fichiers de log
└── checkpoints/               # Modèles sauvegardés
```

## 🔧 Dépannage

### Problèmes Courants

1. **Erreurs d'Import** : Assurez-vous que toutes les dépendances sont installées

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-cli.txt
   ```

2. **Configuration Non Trouvée** : Vérifiez le chemin du fichier config

   ```bash
   python easytpp_cli.py list-configs
   ```

3. **Problèmes CUDA** : Vérifiez la disponibilité du dispositif

   ```bash
   python easytpp_cli.py info
   ```

### Mode Verbeux

Activez le logging verbeux pour le débogage :

```bash
python easytpp_cli.py --verbose run --config config.yaml --experiment THP --dataset H2expc
```

## 📊 Exemples d'Utilisation

### Exemple 1 : Entraînement Rapide

```bash
# Démarrer le mode interactif pour une configuration guidée
python easytpp_cli.py interactive
```

### Exemple 2 : Pipeline Automatisé

```bash
# Entraîner un modèle
python easytpp_cli.py run -c configs/thp_config.yaml -e THP -d taxi -p train --output experiments/taxi_run_001

# Évaluer le modèle entraîné
python easytpp_cli.py run -c configs/thp_config.yaml -e THP -d taxi -p test --checkpoint experiments/taxi_run_001/checkpoints/best.ckpt
```

### Exemple 3 : Gestion de Configuration

```bash
# Lister toutes les configurations disponibles
python easytpp_cli.py list-configs

# Valider une configuration spécifique
python easytpp_cli.py validate -c configs/new_config.yaml -e MyExperiment -d MyDataset

# Exécuter avec validation
python easytpp_cli.py run -c configs/new_config.yaml -e MyExperiment -d MyDataset -p train
```

## 🚀 Fonctionnalités Avancées

### CLI Moderne avec Typer

Utilisez la version moderne du CLI pour une expérience améliorée :

```bash
python easytpp_modern_cli.py run --help
python easytpp_modern_cli.py interactive
```

### Scripts d'Exemple

Explorez les scripts d'exemple dans le dossier `examples/` :

```bash
cd examples
python basic_example.py
python advanced_pipeline.py
```

### Configuration Personnalisée

Créez vos propres templates de configuration :

1. Copiez `configs/runner_config_template.yaml`
2. Modifiez selon vos besoins
3. Utilisez avec le CLI

## 📝 Logging

Le CLI fournit un système de logging professionnel :

- **Niveaux** : DEBUG, INFO, WARNING, ERROR
- **Rotation** : Logs rotatifs quotidiens
- **Rétention** : Conservation configurable
- **Format** : Messages structurés avec horodatage

## 🤝 Contribution

Pour contribuer à l'outil CLI :

1. Forkez le dépôt
2. Créez une branche de fonctionnalité
3. Ajoutez des tests pour les nouvelles fonctionnalités
4. Soumettez une pull request

## 📞 Support

Pour le support et les questions :

- Consultez la documentation
- Ouvrez une issue sur GitHub
- Contactez l'équipe EasyTPP

## 📜 Licence

Ce projet est sous licence Apache 2.0. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

**EasyTPP CLI v2.0** - Rendre la recherche en Processus Ponctuels Temporels accessible et professionnelle.

### 🎯 Prochaines Étapes

1. **Testez le CLI** : `python easytpp_cli.py --help`
2. **Mode Interactif** : `python easytpp_cli.py interactive`
3. **Lisez la doc** : `CLI_README.md`
4. **Exemples** : `cd examples && python basic_example.py`

**Démarrage Rapide** :

```bash
python easytpp_cli.py run --config configs/runner_config_template.yaml --experiment THP --dataset H2expc --phase test
```
