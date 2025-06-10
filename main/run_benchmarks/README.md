# Outil de Benchmark EasyTPP

Cet outil en ligne de commande permet d'exécuter facilement les différents benchmarks sur les datasets configurés dans `bench_config.yaml`.

## Configuration

Le fichier `bench_config.yaml` contient la configuration des datasets disponibles pour les benchmarks. Chaque dataset doit spécifier :

- `data_format` : Format des données (json, csv, etc.)
- `train_dir`, `valid_dir`, `test_dir` : Répertoires des données
- `data_specs` : Spécifications des données (nombre de types d'événements, token de padding, etc.)

## Benchmarks Disponibles

1. **mean** - Mean Inter-Time Benchmark : prédit le temps moyen entre les événements
2. **mark_distribution** - Mark Distribution Benchmark : échantillonne les types d'événements selon la distribution d'entraînement  
3. **intertime_distribution** - Inter-Time Distribution Benchmark : échantillonne les temps inter-événements selon la distribution d'entraînement
4. **last_mark** - Last Mark Benchmark : prédit le dernier type d'événement observé

## Utilisation

### Afficher l'aide
```bash
python run_bench.py --help
```

### Lister les datasets disponibles
```bash
python run_bench.py --list-datasets
```

### Lister les benchmarks disponibles
```bash
python run_bench.py --list-benchmarks
```

### Exécuter un benchmark spécifique sur un dataset spécifique
```bash
python run_bench.py --dataset taxi --benchmark mean
```

### Exécuter tous les benchmarks sur un dataset
```bash
python run_bench.py --dataset taxi --all-benchmarks
```

### Exécuter un benchmark sur tous les datasets
```bash
python run_bench.py --all-datasets --benchmark mean
```

### Exécuter tous les benchmarks sur tous les datasets
```bash
python run_bench.py --all-datasets --all-benchmarks
```

### Options avancées

#### Utiliser une configuration personnalisée
```bash
python run_bench.py --config my_config.yaml --dataset taxi --benchmark mean
```

#### Spécifier un répertoire de sortie personnalisé
```bash
python run_bench.py --output ./my_results --dataset taxi --benchmark mean
```

#### Mode verbeux pour plus de détails
```bash
python run_bench.py --verbose --dataset taxi --benchmark mean
```

## Résultats

Les résultats sont sauvegardés dans le répertoire `./benchmark_results` par défaut, organisés par dataset :

```
benchmark_results/
├── taxi/
│   ├── mean_results.json
│   ├── mark_distribution_results.json
│   ├── intertime_distribution_results.json
│   └── last_mark_results.json
├── taobao/
│   └── ...
└── amazon/
    └── ...
```

Chaque fichier de résultats contient :
- Les métriques calculées (précision, RMSE, F1-score, etc.)
- Le nom du benchmark et du dataset
- Le nombre de types d'événements
- Le temps d'exécution
- Des informations spécifiques au benchmark

## Exemples Pratiques

### Tester rapidement un dataset
```bash
# Lister les datasets pour voir ce qui est disponible
python run_bench.py --list-datasets

# Exécuter le benchmark le plus rapide (mean) sur le dataset test
python run_bench.py --dataset test --benchmark mean
```

### Benchmark complet pour publication
```bash
# Exécuter tous les benchmarks sur tous les datasets
python run_bench.py --all-datasets --all-benchmarks --output ./paper_results
```

### Comparaison de datasets
```bash
# Comparer les performances du benchmark mean sur tous les datasets
python run_bench.py --all-datasets --benchmark mean
```

### Debug et développement
```bash
# Utiliser le mode verbeux pour débugger des problèmes
python run_bench.py --verbose --dataset test --benchmark mean
```

## Gestion des Erreurs

L'outil gère automatiquement les erreurs et continue l'exécution même si certains benchmarks échouent. Les erreurs sont loggées avec des détails sur :

- Le dataset et benchmark concernés
- Le temps d'exécution avant l'échec
- Le message d'erreur détaillé (en mode `--verbose`)

## Extension

Pour ajouter un nouveau benchmark :

1. Créer une classe héritant de `BaseBenchmark` dans le module `easy_tpp.evaluate.benchmarks`
2. Ajouter l'import et l'entrée dans `AVAILABLE_BENCHMARKS` dans `run_bench.py`

Pour ajouter un nouveau dataset :

1. Ajouter l'entrée dans `bench_config.yaml` avec la configuration appropriée
2. S'assurer que les données sont accessibles dans les répertoires spécifiés
