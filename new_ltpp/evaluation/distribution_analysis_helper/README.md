# Distribution Analysis Helper

Outils d'analyse et de comparaison des distributions de processus ponctuels temporels.

## Contenu

Ce package fournit des outils pour :

- **Extraction de données** : Extrait des données depuis diverses sources (datasets, simulations)
- **Analyse statistique** : Analyse complète des distributions avec visualisations
- **Génération de graphiques** : Graphiques de comparaison pour temps inter-événements, types d'événements, longueurs de séquences
- **Calcul de métriques** : Calcul de statistiques de comparaison entre distributions

## Composants principaux

- **Data Extractors** : Extraction de données depuis TPPDataset, DataLoaders, ou résultats de simulation
- **Plot Generators** : Générateurs de graphiques spécialisés pour différents types d'analyse
- **Distribution Analyzer** : Utilitaires d'analyse statistique et de régression
- **Metrics Calculator** : Calcul de métriques de comparaison
- **Comparator** : Orchestrateur principal pour les workflows d'analyse complète

## Utilisation

### Exemple complet

```python
from easy_tpp.evaluate.distribution_analysis_helper import (
    NTPPComparator,
    TPPDatasetExtractor,
    SimulationDataExtractor,
    MetricsCalculatorImpl
)
from easy_tpp.evaluate.distribution_analysis_helper.plot_generators import (
    InterEventTimePlotGenerator,
    EventTypePlotGenerator,
    SequenceLengthPlotGenerator
)

# Initialiser les extracteurs
label_extractor = TPPDatasetExtractor(ground_truth_dataset)
simulation_extractor = SimulationDataExtractor(simulation_results)

# Configurer les générateurs de graphiques
plot_generators = [
    InterEventTimePlotGenerator(),
    EventTypePlotGenerator(), 
    SequenceLengthPlotGenerator()
]

# Initialiser le calculateur de métriques
metrics_calculator = MetricsCalculatorImpl()

# Créer le comparateur et lancer l'analyse
comparator = NTPPComparator(
    label_extractor=label_extractor,
    simulation_extractor=simulation_extractor,
    plot_generators=plot_generators,
    metrics_calculator=metrics_calculator,
    output_dir="./analysis_results"
)

# Les résultats sont automatiquement sauvegardés
```

## Types de graphiques générés

### 1. Distribution des temps inter-événements

- Histogrammes de densité avec échelle logarithmique
- Analyse de régression pour identifier les patterns exponentiels
- Graphiques QQ pour évaluer la similarité des distributions

### 2. Distribution des types d'événements

- Graphiques en barres comparatifs pour chaque type d'événement
- Résumé statistique des 3 types les plus fréquents

### 3. Distribution des longueurs de séquences

- Histogrammes comparatifs normalisés par densité
- Annotations statistiques (moyenne, médiane, écart-type)

### 4. Analyse de corrélation croisée

- Fonction de corrélation croisée entre processus
- Analyse des dépendances temporelles et patterns de clustering

## Fichiers de sortie

L'analyse génère automatiquement :

```bash
output_dir/
├── inter_event_time_comparison.png
├── inter_event_time_comparison_qq.png
├── event_type_distribution.png
├── sequence_length_comparison.png
├── sequence_length_comparison_qq.png
├── cross_correlation_analysis.png
└── metrics_summary.json
```

## Résultats

L'analyse produit :

- **Graphiques de visualisation** : Comparaisons de densités, analyses de distribution, graphiques de corrélation
- **Métriques JSON** : Résumés statistiques complets et métriques de comparaison
- **Analyse de régression** : Analyse des relations statistiques entre distributions
