# Runner-Based Tests Documentation

Ce répertoire contient des tests qui utilisent l'approche runner pour instancier et tester les modèles, similaire à l'approche utilisée dans le dossier `main`.

## Structure des nouveaux tests

### 1. Tests d'intégration (`test_runner_based_integration.py`)
Tests d'intégration complets qui utilisent la configuration YAML et le runner pour :
- Instancier les modèles NHP, RMTPP et Hawkes
- Tester les étapes de training, test et validation
- Vérifier le chargement de checkpoints
- Comparer plusieurs modèles

### 2. Tests unitaires (`test_runner_based_unit.py`)
Tests unitaires pour les composants individuels via le runner :
- Création de modèles via configuration
- Configuration des dataloaders
- Configuration des callbacks PyTorch Lightning
- Gestion des chemins de checkpoints
- Configuration de précision et devices

### 3. Tests fonctionnels (`test_runner_training_steps.py`)
Tests fonctionnels spécifiques aux étapes d'entraînement :
- Exécution des étapes de training
- Exécution des étapes de test
- Exécution des étapes de prédiction
- Validation pendant l'entraînement
- Sauvegarde de checkpoints
- Early stopping
- Reprise d'entraînement depuis checkpoint

## Avantages de cette approche

1. **Réalisme** : Les tests utilisent la même approche que les scripts de production dans `main/`
2. **Configuration complète** : Teste l'ensemble de la pipeline de configuration YAML → Config → Trainer
3. **Flexibilité** : Permet de tester différentes configurations de modèles facilement
4. **Maintenance** : Plus facile à maintenir car suit la même structure que le code de production

## Utilisation

### Exécuter tous les nouveaux tests runner-based :
```bash
pytest tests/integration/test_runner_based_integration.py tests/unit/test_runner_based_unit.py tests/functional/test_runner_training_steps.py -v
```

### Exécuter par catégorie :
```bash
# Tests d'intégration
pytest tests/integration/test_runner_based_integration.py -v

# Tests unitaires
pytest tests/unit/test_runner_based_unit.py -v

# Tests fonctionnels
pytest tests/functional/test_runner_training_steps.py -v
```

### Exécuter avec marqueurs spécifiques :
```bash
# Tests d'intégration seulement
pytest -m integration tests/integration/test_runner_based_integration.py -v

# Tests unitaires seulement
pytest -m unit tests/unit/test_runner_based_unit.py -v

# Tests fonctionnels seulement
pytest -m functional tests/functional/test_runner_training_steps.py -v
```

## Configuration des tests

Les tests utilisent des configurations YAML temporaires qui sont créées dynamiquement pour chaque test. Ces configurations incluent :

- **Données** : Configuration des datasets de test
- **Modèles** : Configuration spécifique pour NHP, RMTPP et HawkesModel
- **Entraînement** : Configuration des paramètres d'entraînement (epochs, patience, etc.)
- **Callbacks** : Configuration des checkpoints et early stopping

## Mocking et isolation

Les tests utilisent extensive mocking pour :
- Éviter les dépendances sur les fichiers de données réels
- Isoler les composants testés
- Accélérer l'exécution des tests
- Permettre des tests déterministes

## Paramètres testés

### Modèles supportés :
- NHP (Neural Hawkes Process)
- RMTPP (Recurrent Marked Temporal Point Process)
- HawkesModel (Classical Hawkes Process)

### Aspects testés :
- Création et initialisation des modèles
- Configuration des dataloaders
- Exécution des étapes d'entraînement
- Validation et test
- Prédictions et simulations
- Gestion des checkpoints
- Configuration des optimiseurs
- Early stopping et callbacks

## Intégration avec les tests existants

Ces nouveaux tests complètent les tests existants sans les remplacer :
- Les tests existants testent l'instanciation directe des modèles
- Les nouveaux tests testent l'approche runner-based
- Ensemble, ils fournissent une couverture complète des deux approches d'utilisation du framework
