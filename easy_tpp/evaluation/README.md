# Evaluation

Outils d'évaluation pour les modèles de processus ponctuels temporels.

## Contenu

Ce dossier contient trois composants principaux :

- **benchmarks/**: Implémentations de base de référence
- **distribution_analysis_helper/**: Analyse et visualisation de distributions
- **metrics_helper/**: Calcul de métriques d'évaluation

## Utilisation rapide

### Évaluation basique

```python
from easy_tpp.evaluation import MetricsHelper, EvaluationMode

# Calculer les métriques pour les prédictions du modèle
helper = MetricsHelper(num_event_types=5, mode=EvaluationMode.PREDICTION)
metrics = helper.compute_all_metrics(batch, predictions)

print(f"Time RMSE: {metrics['time_rmse']:.4f}")
print(f"Type Accuracy: {metrics['type_accuracy']:.4f}")
```

### Exécuter des benchmarks

```python
from easy_tpp.evaluation.benchmarks import LastMarkBenchmark

# Comparer contre une ligne de base
benchmark = LastMarkBenchmark(data_config, "experiment_1")
results = benchmark.evaluate()
```

### Analyse de distribution

```python
from easy_tpp.evaluation.distribution_analysis_helper import TemporalPointProcessComparator

# Analyser les distributions de données
comparator = TemporalPointProcessComparator(
    label_extractor=label_extractor,
    simulation_extractor=simulation_extractor,
    output_dir="./analysis_results"
)
comparator.run_analysis()
```

## Composants

### Benchmarks
Modèles de base simples pour la comparaison de performance :
- **LastMarkBenchmark** : Prédit le type d'événement suivant avec le type précédent
- **MeanInterTimeBenchmark** : Prédit les temps avec la moyenne des données d'entraînement
- **Distribution-based benchmarks** : Échantillonnent depuis les distributions empiriques

### Metrics Helper  
Calcul de métriques d'évaluation pour les modèles TPP :
- **Métriques temporelles** : RMSE, MAE, MAPE
- **Métriques de types** : Précision, F1-score, Precision/Recall
- **Métriques de simulation** : Distance de Wasserstein, MMD

### Distribution Analysis
Outils d'analyse statistique et de visualisation :
- **Extracteurs de données** : Extraction depuis différentes sources
- **Générateurs de graphiques** : Visualisations de comparaison
- **Analyse statistique** : Métriques de comparaison de distributions

## Métriques disponibles

### Métriques basées sur le temps
- RMSE, MAE, MAPE
- Distance de Wasserstein (pour les simulations)

### Métriques basées sur les types  
- Précision de classification
- F1-score (macro/micro)
- Precision/Recall

## Exemples d'utilisation

### Évaluer les performances du modèle

```python
# Initialiser le helper de métriques
helper = MetricsHelper(num_event_types=3)

# Calculer les métriques sur les données de test
for batch in test_loader:
    predictions = model(batch)
    metrics = helper.compute_all_metrics(batch, predictions)
    print(f"Batch metrics: {metrics}")
```

### Comparer contre les baselines

```python
# Exécuter plusieurs benchmarks
benchmarks = [
    LastMarkBenchmark(data_config, "baseline_1"),
    MeanInterTimeBenchmark(data_config, "baseline_2")
]

for benchmark in benchmarks:
    results = benchmark.evaluate()
    print(f"{benchmark.benchmark_name}: {results['metrics']}")
```