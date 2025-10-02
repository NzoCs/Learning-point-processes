# Benchmarks

Implémentations de modèles de base pour l'évaluation des modèles TPP.

## Contenu

Ce module fournit des stratégies de prédiction simples qui servent de références de performance :

- **LastMarkBenchmark** : Prédit le type d'événement suivant en utilisant le type précédent
- **MeanInterTimeBenchmark** : Prédit les temps en utilisant la moyenne des temps d'entraînement
- **MarkDistributionBenchmark** : Échantillonne les types depuis la distribution empirique
- **InterTimeDistributionBenchmark** : Échantillonne les temps depuis la distribution empirique

## Utilisation

### Exemple de base

```python
from easy_tpp.evaluate.benchmarks import LastMarkBenchmark

# Configuration des données
data_config = DataConfig.from_dict(config_dict["data_config"])

# Exécuter le benchmark
benchmark = LastMarkBenchmark(data_config, "my_experiment")
results = benchmark.evaluate()

print(f"Type Accuracy: {results['metrics']['type_accuracy_mean']:.4f}")
```

### Exécuter plusieurs benchmarks

```python
from easy_tpp.evaluate.benchmarks import (
    LastMarkBenchmark, MeanInterTimeBenchmark, 
    MarkDistributionBenchmark, InterTimeDistributionBenchmark
)

benchmarks = [
    LastMarkBenchmark(data_config, "exp1"),
    MeanInterTimeBenchmark(data_config, "exp2"),
    MarkDistributionBenchmark(data_config, "exp3"),
    InterTimeDistributionBenchmark(data_config, "exp4")
]

results = {}
for benchmark in benchmarks:
    results[benchmark.benchmark_name] = benchmark.evaluate()
```

### Utilisation en ligne de commande

```bash
# Benchmark LastMark
python -m easy_tpp.evaluate.benchmarks.last_mark_bench \
    --config_path config.yaml \
    --dataset_name my_dataset \
    --save_dir ./results

# Benchmark MeanInterTime
python -m easy_tpp.evaluate.benchmarks.mean_bench \
    --config_path config.yaml \
    --dataset_name my_dataset \
    --save_dir ./results
```

## Format des résultats

Les résultats sont sauvegardés en JSON :

```json
{
  "benchmark_name": "lag1_mark_benchmark",
  "dataset_name": "dataset_name",
  "num_event_types": 5,
  "metrics": {
    "type_accuracy_mean": 0.6234,
    "type_accuracy_std": 0.0123,
    "macro_f1score_mean": 0.5123,
    "macro_f1score_std": 0.0089
  },
  "num_batches_evaluated": 150
}
```
