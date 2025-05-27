# Tests basés sur le Runner

Ce répertoire contient une nouvelle approche de test qui utilise le système de configuration runner d'EasyTPP, similaire à celui utilisé dans le dossier `main`.

## Vue d'ensemble

Au lieu d'instancier directement chaque modèle individuellement, ces tests utilisent le `Trainer` du runner avec des fichiers de configuration YAML pour instancier et tester les modèles. Cette approche est plus réaliste car elle reproduit exactement la façon dont les modèles sont utilisés en production.

## Structure des nouveaux tests

### 1. Tests d'intégration runner-based (`test_runner_based_integration.py`)

Ces tests vérifient l'intégration complète du système en utilisant le runner :

- **`test_runner_model_instantiation`** : Test l'instanciation des modèles (NHP, RMTPP) via le runner
- **`test_runner_training_step`** : Test l'étape d'entraînement via le runner
- **`test_runner_test_step`** : Test l'étape de test via le runner  
- **`test_runner_predict_step`** : Test l'étape de prédiction via le runner
- **`test_hawkes_model_runner`** : Test spécifique pour le modèle Hawkes
- **`test_config_override_through_runner`** : Test les capacités de surcharge de configuration
- **`test_checkpoint_loading_through_runner`** : Test le chargement de checkpoints
- **`test_multiple_models_comparison_through_runner`** : Test la comparaison de plusieurs modèles

### 2. Tests unitaires runner-based (`test_runner_based_unit.py`)

Ces tests vérifient les composants individuels via le runner :

- **Création de modèles** : `test_runner_model_creation_nhp`, `test_runner_model_creation_rmtpp`, `test_runner_model_creation_hawkes`
- **Création de datamodule** : `test_runner_datamodule_creation`
- **Configuration PyTorch Lightning** : `test_runner_pytorch_lightning_trainer_creation`
- **Configuration des callbacks** : `test_runner_callbacks_configuration`
- **Logique des checkpoints** : `test_runner_checkpoint_path_logic`
- **Configuration d'entraînement** : `test_runner_training_configuration`
- **Précision et dispositifs** : `test_runner_precision_and_device_configuration`
- **Forward pass** : `test_runner_model_forward_pass`
- **Gestion d'erreurs** : `test_runner_error_handling`
- **Configuration du logger** : `test_runner_logger_configuration`
- **Tailles de batch différentes** : `test_runner_different_batch_sizes`

### 3. Tests fonctionnels des étapes d'entraînement (`test_runner_training_steps.py`)

Ces tests vérifient spécifiquement les étapes de training, test et validation :

- **Exécution des étapes d'entraînement** : `test_training_step_execution`
- **Exécution des étapes de test** : `test_testing_step_execution`
- **Exécution des étapes de prédiction** : `test_prediction_step_execution`
- **Étape de validation pendant l'entraînement** : `test_validation_step_during_training`
- **Sauvegarde de checkpoints** : `test_checkpoint_saving_during_training`
- **Arrêt précoce** : `test_early_stopping_during_training`
- **Reprise d'entraînement avec checkpoint** : `test_training_with_checkpoint_resumption`
- **Test avec chargement de checkpoint** : `test_testing_with_checkpoint_loading`
- **Prédiction avec chargement de checkpoint** : `test_prediction_with_checkpoint_loading`
- **Gestion d'erreurs d'entraînement** : `test_training_step_error_handling`
- **Configurations d'optimiseur différentes** : `test_different_optimizer_configurations`
- **Pipeline d'entraînement complet** : `test_complete_training_pipeline`

## Avantages de cette approche

1. **Réalisme** : Les tests utilisent exactement la même approche de configuration que le code de production
2. **Couverture complète** : Test de l'ensemble du pipeline depuis la configuration jusqu'à l'exécution
3. **Facilité de maintenance** : Les configurations de test sont similaires aux configurations réelles
4. **Flexibilité** : Facile d'ajouter de nouveaux modèles ou configurations
5. **Isolation** : Chaque test utilise des configurations temporaires isolées

## Configuration des tests

Les tests utilisent des configurations YAML minimales créées dynamiquement avec les caractéristiques suivantes :

### Configuration de base
```yaml
pipeline_config_id: runner_config
data:
  test_data:
    data_format: json
    train_dir: test/data
    valid_dir: test/data
    test_dir: test/data
    data_specs:
      num_event_types: 2
      pad_token_id: 2
      padding_side: left
```

### Configuration NHP
```yaml
NHP_test:
  data_loading_specs:
    batch_size: 4
    num_workers: 1
  model_config:
    model_id: NHP
    specs:
      hidden_size: 16
      time_emb_size: 8
      num_layers: 1
    thinning:
      num_exp: 10
      over_sample_rate: 1.5
      dtime_max: 5
      num_sample: 5
    base_config:
      lr: 0.01
      lr_scheduler: false
  trainer_config:
    stage: train
    max_epochs: 1
    val_freq: 1
    accumulate_grad_batches: 1
    patience: 3
    devices: 1
    use_precision_16: false
    log_freq: 1
    checkpoints_freq: 1
```

### Configuration RMTPP
Similaire à NHP avec des paramètres adaptés.

### Configuration Hawkes
```yaml
Hawkes_test:
  model_config:
    model_id: HawkesModel
    specs:
      mu: [0.1]
      alpha: [[0.2]]
      beta: [[1.0]]
    thinning:
      num_exp: 10
      num_sample: 5
      over_sample_rate: 1.5
      dtime_max: 5
```

## Exécution des tests

Pour exécuter tous les nouveaux tests runner-based :

```bash
# Tests d'intégration
pytest tests/integration/test_runner_based_integration.py -v

# Tests unitaires
pytest tests/unit/test_runner_based_unit.py -v

# Tests fonctionnels
pytest tests/functional/test_runner_training_steps.py -v

# Tous les nouveaux tests
pytest tests/integration/test_runner_based_integration.py tests/unit/test_runner_based_unit.py tests/functional/test_runner_training_steps.py -v

# Avec un modèle spécifique
pytest tests/integration/test_runner_based_integration.py::TestRunnerBasedIntegration::test_runner_model_instantiation[NHP] -v
```

## Mocking et isolation

Les tests utilisent des mocks pour :

- **Data loading** : `easy_tpp.preprocess.data_loader.TPPDataModule.setup`
- **Data loaders** : `train_dataloader`, `val_dataloader`, `test_dataloader`
- **PyTorch Lightning** : `Trainer.fit`, `Trainer.test`, `Trainer.predict`
- **File operations** : `builtins.open`, `json.dump`
- **Comparators** : `easy_tpp.evaluate.new_comparator.NewDistribComparator`

Cette approche garantit que les tests sont rapides et isolés, sans dépendances sur les fichiers de données réels.

## Comparaison avec les tests existants

| Aspect | Tests existants | Tests runner-based |
|--------|----------------|-------------------|
| **Instanciation** | Directe via classes | Via runner et configuration YAML |
| **Configuration** | Objets Python | Fichiers YAML temporaires |
| **Réalisme** | Modéré | Élevé (production-like) |
| **Maintenance** | Modifications dans le code | Modifications dans la configuration |
| **Couverture** | Composants individuels | Pipeline complet |

## Extension des tests

Pour ajouter un nouveau modèle aux tests :

1. Ajouter la configuration du modèle dans les méthodes `_create_*_config`
2. Ajouter des tests paramétrés avec le nouveau modèle
3. Adapter les assertions selon les spécificités du modèle

Exemple pour un nouveau modèle `NewModel` :

```python
def _create_minimal_config(self, model_id='NHP', temp_dir=None):
    # ... existing code ...
    elif model_id == 'NewModel':
        specs = {
            'param1': value1,
            'param2': value2
        }
    # ... rest of the code ...

@pytest.mark.parametrize("model_id", ["NHP", "RMTPP", "NewModel"])
def test_runner_model_instantiation(self, temporary_directory, model_id):
    # Test will automatically include NewModel
```

## Notes importantes

1. **Fichiers temporaires** : Tous les tests utilisent des répertoires temporaires pour éviter les conflits
2. **Seed fixe** : `set_seed(42)` est utilisé pour la reproductibilité
3. **Mocking complet** : Aucune dépendance sur des fichiers de données réels
4. **Isolation** : Chaque test est complètement isolé des autres
5. **Performance** : Tests rapides grâce au mocking approprié

Cette approche fournit une couverture de test plus réaliste et maintenable pour le système EasyTPP.
