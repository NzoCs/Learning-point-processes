# Runner

Module d'exécution et d'entraînement des modèles TPP.

## Contenu

Ce dossier contient les outils pour orchestrer l'exécution des expériences et l'entraînement des modèles :

- **runner.py** : Orchestrateur principal pour l'exécution des expériences complètes
- **trainer.py** : Module d'entraînement des modèles avec PyTorch Lightning

## Utilisation

### Runner - Orchestrateur d'expériences

Le `Runner` coordonne tous les aspects d'une expérience : données, modèle, entraînement, évaluation.

```python
from easy_tpp.runner import Runner
from easy_tpp.config_factory import config_factory

# Charger la configuration
config = config_factory.from_yaml('experiment_config.yaml')

# Créer et lancer l'expérience
runner = Runner(config)
runner.run()
```

### Configuration d'une expérience

```yaml
# experiment_config.yaml
data_config:
  data_dir: "./data/synthetic"
  data_format: "json"
  data_specs:
    num_event_types: 5
    max_seq_len: 100
  data_loading_specs:
    batch_size: 64
    shuffle: true

model_config:
  model_type: "NHP"
  model_specs:
    hidden_size: 128
    num_layers: 2
    dropout: 0.1

runner_config:
  dataset_id: "synthetic_experiment"
  model_id: "nhp_baseline"
  max_epochs: 100
  patience: 20
  val_freq: 5
  save_dir: "./experiments/"
```

### Phases d'exécution

Le Runner supporte différentes phases d'exécution :

```python
# Entraînement seulement
runner.run(phase='train')

# Évaluation seulement
runner.run(phase='test')

# Prédiction
runner.run(phase='predict')

# Pipeline complet
runner.run(phase='all')  # train -> validate -> test
```

### Trainer - Entraînement avec PyTorch Lightning

Le `Trainer` encapsule la logique d'entraînement avec PyTorch Lightning.

```python
from easy_tpp.runner.trainer import Trainer
from easy_tpp.models import NHP
from easy_tpp.data.preprocess import TPPDataModule

# Créer les composants
model = NHP(hidden_size=128, num_event_types=5)
data_module = TPPDataModule(data_config)
trainer = Trainer(
    max_epochs=100,
    patience=20,
    val_check_interval=0.2
)

# Entraîner le modèle
trainer.fit(model, datamodule=data_module)
```

## Workflow d'expérience complet

### 1. Préparation des données

```python
# Le Runner configure automatiquement les données
data_module = runner.setup_data()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

### 2. Initialisation du modèle

```python
# Création automatique basée sur la configuration
model = runner.setup_model()
print(f"Model type: {model.__class__.__name__}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
```

### 3. Configuration de l'entraînement

```python
# Configuration des callbacks et logger
trainer = runner.setup_trainer()

# Callbacks inclus automatiquement :
# - EarlyStopping
# - ModelCheckpoint
# - ProgressBar
# - Logger (WandB, TensorBoard, etc.)
```

### 4. Entraînement

```python
# Lancement de l'entraînement
trainer.fit(model, datamodule=data_module)

# Restauration du meilleur modèle
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Best model saved at: {best_model_path}")
```

### 5. Évaluation

```python
# Évaluation sur le test set
test_results = trainer.test(model, datamodule=data_module)
print(f"Test results: {test_results}")

# Métriques détaillées
from easy_tpp.evaluation import MetricsHelper

helper = MetricsHelper(num_event_types=5)
detailed_metrics = helper.evaluate_model(model, test_loader)
```

## Configuration avancée du Runner

### Logging et suivi d'expériences

```python
# Configuration WandB
logger_config = {
    'logger_type': 'wandb',
    'project': 'tpp_experiments',
    'name': 'nhp_experiment_1',
    'tags': ['baseline', 'nhp']
}

# Configuration TensorBoard
logger_config = {
    'logger_type': 'tensorboard',
    'save_dir': './logs/',
    'name': 'nhp_logs'
}
```

### Callbacks personnalisés

```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# Configuration des callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min'
    ),
    ModelCheckpoint(
        monitor='val_loss',
        save_top_k=3,
        mode='min',
        filename='nhp-{epoch:02d}-{val_loss:.2f}'
    ),
    LearningRateMonitor(logging_interval='step')
]

runner.setup_trainer(callbacks=callbacks)
```

### Optimisation des hyperparamètres

```python
# Intégration avec Optuna
from easy_tpp.runner import HPORunner

def objective(trial):
    # Suggérer des hyperparamètres
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    # Mettre à jour la configuration
    config.model_config.model_specs['hidden_size'] = hidden_size
    config.runner_config.training_specs['learning_rate'] = lr
    
    # Exécuter l'expérience
    runner = Runner(config)
    results = runner.run()
    
    return results['val_loss']

# Lancer l'optimisation
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

## Monitoring et debugging

### Surveillance de l'entraînement

```python
# Activation du mode debug
runner = Runner(config, debug=True)

# Logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Profilage des performances
trainer = runner.setup_trainer(profiler='simple')
```

### Gestion des checkpoints

```python
# Reprendre l'entraînement depuis un checkpoint
runner = Runner(config)
runner.resume_from_checkpoint('path/to/checkpoint.ckpt')

# Sauvegarder manuellement
runner.save_checkpoint('manual_checkpoint.ckpt')

# Charger un modèle pré-entraîné
model = runner.load_model_from_checkpoint('best_model.ckpt')
```

### Validation croisée

```python
# K-fold cross validation
from easy_tpp.runner import CrossValidationRunner

cv_runner = CrossValidationRunner(config, k_folds=5)
cv_results = cv_runner.run()

print(f"CV Mean: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
```

## Gestion des erreurs et récupération

### Gestion des échecs d'entraînement

```python
try:
    runner = Runner(config)
    results = runner.run()
except Exception as e:
    print(f"Training failed: {e}")
    
    # Diagnostic automatique
    runner.diagnose_failure()
    
    # Tentative de récupération
    runner.recover_and_retry()
```

### Nettoyage automatique

```python
# Configuration du nettoyage
cleanup_config = {
    'remove_failed_experiments': True,
    'keep_best_n_checkpoints': 3,
    'cleanup_temp_files': True
}

runner = Runner(config, cleanup_config=cleanup_config)
```

## Exemples d'utilisation complète

### Expérience simple

```python
from easy_tpp.runner import Runner
from easy_tpp.config_factory import config_factory

# Configuration minimale
config = config_factory.from_dict({
    'data_config': {
        'data_dir': './data/',
        'data_format': 'json'
    },
    'model_config': {
        'model_type': 'NHP',
        'model_specs': {'hidden_size': 64}
    },
    'runner_config': {
        'max_epochs': 50
    }
})

# Exécution
runner = Runner(config)
results = runner.run()
print(f"Final validation loss: {results['val_loss']:.4f}")
```

### Expérience avec comparaison de modèles

```python
models_to_compare = ['NHP', 'THP', 'RMTPP']
results = {}

for model_type in models_to_compare:
    print(f"Training {model_type}...")
    
    config.model_config.model_type = model_type
    runner = Runner(config)
    
    model_results = runner.run()
    results[model_type] = model_results
    
    print(f"{model_type} - Val Loss: {model_results['val_loss']:.4f}")

# Meilleur modèle
best_model = min(results.keys(), key=lambda k: results[k]['val_loss'])
print(f"Best model: {best_model}")
```

### Pipeline de production

```python
def production_pipeline(config_path):
    """Pipeline complet pour la production"""
    
    # 1. Charger la configuration
    config = config_factory.from_yaml(config_path)
    
    # 2. Validation de la configuration
    config.validate()
    
    # 3. Entraînement
    runner = Runner(config)
    results = runner.run(phase='train')
    
    # 4. Évaluation
    test_results = runner.run(phase='test')
    
    # 5. Sauvegarde du modèle final
    runner.export_model('production_model.pt')
    
    # 6. Génération du rapport
    runner.generate_report('experiment_report.html')
    
    return results, test_results

# Utilisation
train_results, test_results = production_pipeline('production_config.yaml')
```
