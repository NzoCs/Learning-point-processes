#!/usr/bin/env python3
"""
Exemple simple d'entraînement d'un modèle Temporal Point Process

Ce script montre comment :
1. Configurer un modèle TPP
2. Charger des données
3. Entraîner le modèle
4. Évaluer les performances

Usage:
    python train_example.py
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire racine du projet au Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config
from easy_tpp.utils import logger


def train_simple_model() -> bool:
    """
    Exemple d'entraînement d'un modèle Neural Hawkes Process (NHP) sur le dataset test.
    """

    # === 1. CONFIGURATION ===
    print("🔧 Configuration du modèle...")

    # Chemin vers le fichier de configuration
    config_path = Path(__file__).parent / "runner_config.yaml"

    # ID de l'expérience et du dataset (définis dans runner_config.yaml)
    experiment_id = "NHP"  # Modèle Neural Hawkes Process
    dataset_id = "test"  # Dataset de test simple

    # Construire la configuration à partir du YAML
    config_dict = parse_runner_yaml_config(str(config_path), experiment_id, dataset_id)

    # Créer l'objet de configuration
    config = RunnerConfig.from_dict(config_dict)

    print(f"✅ Modèle: {config.model_config.model_id}")
    print(f"✅ Dataset: {dataset_id}")
    print(f"✅ Batch size: {config.data_loading_specs.batch_size}")

    # === 2. INITIALISATION DU RUNNER ===
    print("\n🚀 Initialisation du runner...")

    # Créer le répertoire de sortie
    output_dir = "./simple_experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    # Initialiser le runner
    runner = Runner(config=config, output_dir=output_dir)

    print(f"✅ Runner initialisé, résultats sauvés dans: {output_dir}")

    # === 3. ENTRAÎNEMENT ===
    print("\n🎓 Démarrage de l'entraînement...")

    try:
        # Entraîner le modèle
        runner.run(phase="train")
        print("✅ Entraînement terminé avec succès!")

        # Évaluer sur les données de test
        runner.run(phase="test")
        print("✅ Évaluation terminée avec succès!")

        # Générer les prédictions et comparaisons de distributions
        runner.run(phase="predict")
        print("✅ Prédictions et analyses de distribution générées!")

    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        return False

    print(f"\n🎉 Expérience terminée! Résultats dans: {output_dir}")
    return True


def train_hawkes_model() -> bool:
    """
    Exemple d'entraînement d'un modèle Hawkes classique.
    Plus simple car les paramètres sont fixes (pas d'apprentissage neural).
    """

    print("🔧 Configuration du modèle Hawkes...")

    config_path = Path(__file__).parent / "runner_config.yaml"
    experiment_id = "hawkes1"  # Modèle Hawkes univarié
    dataset_id = "hawkes1"  # Dataset correspondant

    config_dict = parse_runner_yaml_config(str(config_path), experiment_id, dataset_id)

    config = RunnerConfig.from_dict(config_dict)

    print(f"✅ Modèle: {config.model_config.model_id}")
    print(f"✅ Paramètres Hawkes:")
    print(f"   - mu (intensité de base): {config.model_config.specs.mu}")
    print(f"   - alpha (influence): {config.model_config.specs.alpha}")
    print(f"   - beta (décroissance): {config.model_config.specs.beta}")

    output_dir = "./hawkes_experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    runner = Runner(config=config, output_dir=output_dir)

    print("\n🎯 Test du modèle Hawkes...")
    try:
        # Pour Hawkes, on fait directement le test (pas d'entraînement nécessaire)
        runner.run(phase="test")
        print("✅ Test terminé avec succès!")

        # Générer prédictions et comparaisons avec données réelles
        runner.run(phase="predict")
        print("✅ Prédictions Hawkes et comparaisons générées!")

        print(f"📁 Résultats dans: {output_dir}")
        return True

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def main() -> int:
    """
    Fonction principale - choisir quel exemple exécuter.
    """

    print("=== EXEMPLES D'ENTRAÎNEMENT EasyTPP ===\n")

    print("Choisissez un exemple:")
    print("1. Modèle Neural Hawkes Process (NHP) - avec entraînement")
    print("2. Modèle Hawkes classique - test direct")
    print("3. Les deux")

    choice = input("\nVotre choix (1/2/3): ").strip()

    if choice == "1":
        print("\n" + "=" * 50)
        print("EXEMPLE 1: Neural Hawkes Process (NHP)")
        print("=" * 50)
        success = train_simple_model()

    elif choice == "2":
        print("\n" + "=" * 50)
        print("EXEMPLE 2: Modèle Hawkes classique")
        print("=" * 50)
        success = train_hawkes_model()

    elif choice == "3":
        print("\n" + "=" * 50)
        print("EXEMPLE 1: Neural Hawkes Process (NHP)")
        print("=" * 50)
        success1 = train_simple_model()

        print("\n" + "=" * 50)
        print("EXEMPLE 2: Modèle Hawkes classique")
        print("=" * 50)
        success2 = train_hawkes_model()

        success = success1 and success2

    else:
        print("❌ Choix invalide!")
        return 1

    if success:
        print("\n🎉 Tous les exemples ont réussi!")
        return 0
    else:
        print("\n❌ Certains exemples ont échoué.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
