# 🚀 EasyTPP Model Runner

Outil en ligne de commande simple pour exécuter des expériences de Temporal Point Process avec EasyTPP.

## 📋 Utilisation

### Commandes de base

```bash
# Lister les expériences disponibles
python run_model.py --list-experiments

# Lister les datasets disponibles
python run_model.py --list-datasets

# Afficher l'aide
python run_model.py --help
```

### Exécuter des expériences

```bash
# Exécuter une expérience spécifique en mode test
python run_model.py --experiment THP --dataset H2expc --phase test

# Exécuter en mode entraînement
python run_model.py --experiment THP --dataset H2expc --phase train

# Exécuter avec un checkpoint
python run_model.py --experiment THP --dataset H2expc --phase test --checkpoint path/to/model.ckpt

# Exécuter toutes les phases
python run_model.py --experiment THP --dataset H2expc --phase all

# Exécuter toutes les expériences sur un dataset
python run_model.py --all-experiments --dataset H2expc --phase test
```

### Options avancées

```bash
# Spécifier un fichier de configuration personnalisé
python run_model.py --config custom_config.yaml --experiment THP --dataset H2expc

# Spécifier un répertoire de sortie
python run_model.py --experiment THP --dataset H2expc --output-dir ./my_results

# Mode verbeux pour plus de détails
python run_model.py --experiment THP --dataset H2expc --verbose

# Mode debug avec traceback complet
python run_model.py --experiment THP --dataset H2expc --debug
```

## ⚙️ Configuration

Le script utilise un fichier de configuration YAML (par défaut `runner_config.yaml`) qui doit contenir :

```yaml
experiments:
  THP:
    model_config:
      # Configuration du modèle
    data_config:
      # Configuration des données
    training_config:
      # Configuration d'entraînement

datasets:
  H2expc:
    # Configuration du dataset
```

## 📊 Phases d'exécution

- **train** : Entraînement du modèle
- **test** : Évaluation du modèle
- **predict** : Génération de prédictions
- **validation** : Validation croisée
- **all** : Exécution de toutes les phases

## 📁 Structure des résultats

Les résultats sont sauvegardés dans la structure suivante :
```
experiment_results/
├── THP_H2expc_train/
├── THP/H2expc_test/
└── ...
```

## 🛠️ Dépannage

- Vérifiez que le fichier de configuration existe
- Assurez-vous que l'expérience et le dataset spécifiés existent dans la configuration
- Utilisez `--verbose` ou `--debug` pour plus d'informations sur les erreurs

## 📝 Exemples

```bash
# Workflow complet
python run_model.py --list-experiments
python run_model.py --list-datasets
python run_model.py --experiment THP --dataset H2expc --phase train
python run_model.py --experiment THP --dataset H2expc --phase test --checkpoint experiment_results/THP_H2expc_train/checkpoints/best.ckpt
```
