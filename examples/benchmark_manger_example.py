"""
Exemple d'utilisation de la BenchmarkFactory avec l'enum

Cet exemple montre comment utiliser la factory pour simplifier
le lancement des benchmarks par rapport au code précédent.
"""

from easy_tpp.configs import DataConfig
from easy_tpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarkManager,
    Benchmarks,
)


def example_simple_benchmark():
    """Exemple simple avec un seul benchmark."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    factory = BenchmarkManager(data_config, "test_dataset")
    results = factory.run_single(Benchmarks.MEAN_INTER_TIME)
    print(f"Résultats: {results}")


def example_multiple_benchmarks():
    """Exemple avec plusieurs benchmarks."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    # Utiliser la factory pour plusieurs benchmarks
    factory = BenchmarkManager(data_config, "test_dataset")

    # Lancer plusieurs benchmarks spécifiques
    selected_benchmarks = [
        Benchmarks.MEAN_INTER_TIME,
        Benchmarks.MARK_DISTRIBUTION,
        Benchmarks.INTERTIME_DISTRIBUTION,
    ]

    results = factory.run_multiple(selected_benchmarks)

    for benchmark_name, result in results.items():
        print(f"Résultats pour {benchmark_name}: terminé")


def example_all_benchmarks():
    """Exemple pour lancer tous les benchmarks."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    factory = BenchmarkManager(data_config, "test_dataset")
    results = factory.run_all()
    print(f"Tous les benchmarks terminés: {len(results)} benchmarks")


def example_by_names():
    """Exemple pour lancer des benchmarks par leurs noms."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    factory = BenchmarkManager(data_config, "test_dataset")

    # Lancer par noms (pratique pour la configuration)
    benchmark_names = ["mean_inter_time", "mark_distribution_sampling"]
    results = factory.run_by_names(benchmark_names)

    print(f"Benchmarks lancés: {list(results.keys())}")


def example_with_parameters():
    """Exemple avec des paramètres personnalisés."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    factory = BenchmarkManager(data_config, "test_dataset", save_dir="./custom_results")

    # Lancer avec des paramètres spécifiques
    results = factory.run_single(
        Benchmarks.INTERTIME_DISTRIBUTION,
        num_bins=100,  # Paramètre personnalisé pour ce benchmark
    )

    print("Benchmark avec paramètres personnalisés terminé")


def main():
    """Fonction principale pour tester tous les exemples."""
    print("=== Exemple simple ===")
    example_simple_benchmark()

    print("\n=== Exemple multiple ===")
    example_multiple_benchmarks()

    print("\n=== Exemple tous benchmarks ===")
    example_all_benchmarks()

    print("\n=== Exemple par noms ===")
    example_by_names()

    print("\n=== Exemple avec paramètres ===")
    example_with_parameters()

    print("\n=== Liste des benchmarks disponibles ===")
    print("Benchmarks disponibles:")
    for benchmark in Benchmarks:
        print(f"- {benchmark.benchmark_name}")


if __name__ == "__main__":
    main()
