# Metrics Helper

Système de calcul de métriques pour les processus ponctuels temporels.

## Contenu

Ce package fournit un système modulaire pour calculer les métriques d'évaluation :

- **MetricsHelper** : Orchestrateur principal pour le calcul de métriques
- **PredictionMetricsComputer** : Calcul des métriques de prédiction
- **SimulationMetricsComputer** : Calcul des métriques de simulation
- **Interfaces** : Interfaces abstraites pour l'extensibilité
- **Types partagés** : Structures de données communes et conteneurs

## Utilisation

### Utilisation de base

```python
from new_ltpp.evaluate.metrics_helper import MetricsHelper, EvaluationMode

# Initialiser pour les métriques de prédiction
helper = MetricsHelper(
    num_event_types=5,
    mode=EvaluationMode.PREDICTION
)

# Calculer toutes les métriques
all_metrics = helper.compute_all_metrics(batch, pred)

# Calculer seulement les métriques temporelles
time_metrics = helper.compute_all_time_metrics(batch, pred)

# Calculer seulement les métriques de type
type_metrics = helper.compute_all_type_metrics(batch, pred)
```

### Calcul sélectif de métriques

```python
from new_ltpp.evaluate.metrics_helper import MetricsHelper, PredictionMetrics

# Initialiser avec seulement les métriques sélectionnées
helper = MetricsHelper(
    num_event_types=5,
    mode=EvaluationMode.PREDICTION,
    selected_prediction_metrics=[
        PredictionMetrics.TIME_RMSE,
        PredictionMetrics.TIME_MAE,
        PredictionMetrics.TYPE_ACCURACY
    ]
)

# Seules les métriques sélectionnées seront calculées
metrics = helper.compute_all_metrics(batch, pred)
```

### Changement de mode

```python
# Basculer entre modes prédiction et simulation
helper.set_mode(EvaluationMode.SIMULATION)
sim_metrics = helper.compute_all_metrics(batch, pred)

helper.set_mode(EvaluationMode.PREDICTION)
pred_metrics = helper.compute_all_metrics(batch, pred)
```

## Métriques disponibles

### Métriques de prédiction

**Métriques temporelles :**
- `time_rmse` : Root Mean Square Error pour les prédictions temporelles
- `time_mae` : Mean Absolute Error pour les prédictions temporelles

**Métriques de type :**
- `type_accuracy` : Précision de classification pour les types d'événements
- `macro_f1score` : F1-score macro-moyenné
- `micro_f1score` : F1-score micro-moyenné
- `precision` : Score de précision
- `recall` : Score de rappel

### Métriques de simulation

**Métriques temporelles :**
- `wasserstein_1d` : Distance de Wasserstein entre séquences temporelles
- `mmd_rbf_padded` : Maximum Mean Discrepancy avec noyau RBF
- `mmd_wasserstein` : MMD avec noyau basé sur Wasserstein

## Exemple d'évaluation complète

```python
# Initialiser le helper de métriques
helper = MetricsHelper(num_event_types=3)

# Évaluer sur les données de test
for batch in test_loader:
    predictions = model(batch)
    metrics = helper.compute_all_metrics(batch, predictions)
    print(f"Batch metrics: {metrics}")
```
