#!/usr/bin/env python3
"""
Exemple MINIMAL d'utilisation d'EasyTPP

Cet exemple montre la façon la plus simple de:
1. Charger une configuration existante
2. Lancer un entraînement/test

Usage:
    python minimal_example.py
    python minimal_example.py --experiment NHP --dataset test
    python minimal_example.py --experiment hawkes1 --dataset hawkes1 --phase train
"""

import sys
import argparse
from pathlib import Path

# Ajouter le répertoire racine du projet au Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root)) # pourquoi ? 

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def run_minimal_example(experiment_id: str = "hawkes1", dataset_id: str = "hawkes1", phase: str = "all") -> None:
    """Exemple minimal - 3 lignes de code principal!"""
    
    print(f"📋 Expérience: {experiment_id}")
    print(f"📊 Dataset: {dataset_id}")
    print(f"⚙️  Phase: {phase}")
    
    # 1. Charger la configuration depuis le YAML
    config_dict = parse_runner_yaml_config(
        yaml_path="./runner_config.yaml",
        experiment_id=experiment_id,
        dataset_id=dataset_id
    )

    # 2. Créer l'objet de configuration
    config = RunnerConfig.from_dict(config_dict)

    # 3. Créer le runner
    runner = Runner(config=config, output_dir="./minimal_results")

    # 4. Lancer l'expérience
    runner.run(phase=phase)
    
    if phase == "all":
        print("✅ Pipeline complet terminé (train → test → predict)!")
        print("   📊 Métriques de performance calculées")
        print("   🔮 Prédictions générées et comparées aux données réelles")
        print("   📈 Analyses de distribution disponibles")
    else:
        print(f"✅ Phase '{phase}' terminée!")
    
    print("📁 Résultats dans ./minimal_results")


def main() -> None:
    """Parse arguments et lance l'exemple."""
    parser = argparse.ArgumentParser(
        description="Exemple minimal EasyTPP avec arguments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Utiliser les valeurs par défaut (hawkes1, phase all)
  python minimal_example.py
  
  # Choisir une expérience spécifique avec pipeline complet
  python minimal_example.py --experiment NHP --dataset test
  python minimal_example.py --experiment THP --dataset H2expc
  
  # Lancer seulement l'entraînement
  python minimal_example.py --experiment NHP --dataset test --phase train
  
  # Lister les options disponibles
  python minimal_example.py --list-experiments
  python minimal_example.py --list-datasets
        """
    )
    
    # Arguments principaux
    parser.add_argument('--experiment', '-e', type=str, default='hawkes1',
                        help='ID de l\'expérience à lancer (défaut: hawkes1)')
    
    parser.add_argument('--dataset', '-d', type=str, default='hawkes1',
                        help='ID du dataset à utiliser (défaut: hawkes1)')
    
    parser.add_argument('--phase', '-p', type=str, default='all',
                        choices=['train', 'test', 'predict', 'all'],
                        help='Phase à exécuter (défaut: all = train+test+predict)')
    
    # Options d'aide
    parser.add_argument('--list-experiments', action='store_true',
                        help='Lister toutes les expériences disponibles')
    
    parser.add_argument('--list-datasets', action='store_true',
                        help='Lister tous les datasets disponibles')
    
    args = parser.parse_args()
    
    # Traiter les options de listing
    if args.list_experiments:
        print("📋 Expériences disponibles dans runner_config.yaml:")
        experiments = [
            "hawkes1", "hawkes2", "H2expi", "H2expc", "test_HawkesModel",
            "NHP", "RMTPP", "AttNHP", "SAHP", "THP", "FullyNN", "IntensityFree"
        ]
        for exp in experiments:
            print(f"  - {exp}")
        return
    
    if args.list_datasets:
        print("📊 Datasets disponibles dans runner_config.yaml:")
        datasets = [
            "taxi", "taobao", "amazon", "test", "H2expi", "H2expc",
            "hawkes1", "hawkes2", "self_correcting"
        ]
        for dataset in datasets:
            print(f"  - {dataset}")
        return
    
    # Lancer l'exemple avec les arguments fournis
    run_minimal_example(args.experiment, args.dataset, args.phase)


if __name__ == "__main__":
    main()
