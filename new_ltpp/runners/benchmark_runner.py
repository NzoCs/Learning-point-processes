"""
Benchmark Runner

Runner pour les tests de performance et benchmarking TPP.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .cli_base import CONFIG_MAP, CLIRunnerBase

try:
    from new_ltpp.configs import DataConfigBuilder
    from new_ltpp.evaluation.benchmarks.benchmark_manager import (
        BenchmarkManager,
    )
    from new_ltpp.evaluation.benchmarks.benchmark_manager import (
        BenchmarksEnum as Benchmarks,
    )
except ImportError as e:
    DataConfigBuilder = None
    BenchmarkManager = None
    Benchmarks = None
    IMPORT_ERROR = str(e)


class BenchmarkRunner(CLIRunnerBase):
    """
    Runner pour les tests de performance et benchmarking.
    Mesure les temps d'exécution, utilisation mémoire, et performance des modèles.
    """

    def __init__(self, debug: bool = False):
        super().__init__("BenchmarkRunner", debug=debug)

    def run_benchmark(
        self,
        config_path: str = None,
        data_config: Union[str, List[str]] = "test",
        data_loading_config: str = "quick_test",
        benchmarks: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        run_all: bool = False,
        run_all_configs: bool = False,
        **benchmark_params,
    ) -> bool:
        """
        Lance des benchmarks TPP en utilisant le BenchmarkManager.

        Args:
            config_path: Chemin vers le fichier YAML de configuration
            data_config: Configuration(s) des données (ex: 'test' ou ['test', 'large'])
            data_loading_config: Configuration du chargement des données
            benchmarks: Liste des noms de benchmarks à exécuter
            output_dir: Répertoire de sortie
            run_all: Exécuter tous les benchmarks disponibles
            run_all_configs: Exécuter sur toutes les configurations
            **benchmark_params: Paramètres supplémentaires pour les benchmarks

        Returns:
            True si les benchmarks se sont déroulés avec succès
        """
        # Vérifier les dépendances
        required_modules = ["new_ltpp.configs", "new_ltpp.evaluation.benchmarks"]
        if not self.check_dependencies(required_modules):
            return False

        try:
            self.print_info("Configuration du benchmark...")

            # Configuration par défaut si aucun fichier spécifié
            if config_path is None:
                config_path = str(self.get_config_path())
                self.print_info(
                    f"Utilisation de la configuration par défaut: {config_path}"
                )

            # Créer le répertoire de sortie si nécessaire
            if output_dir is None:
                output_dir = str(self.get_output_path("benchmarks"))
            else:
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Support de plusieurs configurations
            data_configs_list = (
                data_config if isinstance(data_config, list) else [data_config]
            )

            # Construire toutes les configurations
            all_data_configs = []
            for data_cfg in data_configs_list:
                # Construire les chemins de configuration avec la méthode utilitaire
                config_paths = self._build_config_paths(
                    data=data_cfg, data_loading=data_loading_config
                )

                self.print_info(
                    f"Configuration des données: {config_paths.get('data_config_path')}"
                )

                builder = DataConfigBuilder()
                builder.load_from_yaml(yaml_path=config_path, **config_paths)
                built_config = builder.build()
                all_data_configs.append(built_config)

                self.print_info(f"Configuration chargée: {built_config.dataset_id}")

            # Créer le BenchmarkManager avec une ou plusieurs configs
            if len(all_data_configs) == 1:
                benchmark_manager = BenchmarkManager(all_data_configs[0])
            else:
                benchmark_manager = BenchmarkManager(all_data_configs)
                self.print_info(
                    f"BenchmarkManager créé avec {len(all_data_configs)} configurations"
                )

            # Déterminer quels benchmarks exécuter
            if run_all and run_all_configs and benchmark_manager.get_config_count() > 1:
                self.print_info(
                    f"Exécution de tous les benchmarks sur {benchmark_manager.get_config_count()} configurations..."
                )
                self.print_info(
                    f"Datasets: {', '.join(benchmark_manager.list_datasets())}"
                )
                results = benchmark_manager.run_all_on_all_configs(**benchmark_params)

            elif run_all:
                self.print_info("Exécution de tous les benchmarks disponibles...")
                if benchmark_manager.get_config_count() > 1:
                    results = benchmark_manager.run_all(**benchmark_params)
                else:
                    results = benchmark_manager.run_all(**benchmark_params)

            elif benchmarks:
                self.print_info(f"Exécution des benchmarks: {benchmarks}")

                # Convertir les noms en enum si nécessaire
                benchmark_enums = []
                for bench_name in benchmarks:
                    try:
                        # Essayer de trouver l'enum correspondant
                        benchmark_enum = next(
                            b
                            for b in Benchmarks
                            if b.benchmark_name.lower() == bench_name.lower()
                        )
                        benchmark_enums.append(benchmark_enum)
                    except StopIteration:
                        self.print_error(f"Benchmark non trouvé: {bench_name}")
                        continue

                if benchmark_enums:
                    if run_all_configs and benchmark_manager.get_config_count() > 1:
                        self.print_info(
                            f"Exécution sur {benchmark_manager.get_config_count()} configurations..."
                        )
                        results = benchmark_manager.run_multiple_on_all_configs(
                            benchmark_enums, **benchmark_params
                        )
                    else:
                        results = benchmark_manager.run_multiple(
                            benchmark_enums, **benchmark_params
                        )
                else:
                    self.print_error("Aucun benchmark valide spécifié")
                    return False

            else:
                # Par défaut, exécuter quelques benchmarks essentiels
                self.print_info("Exécution des benchmarks par défaut...")
                default_benchmarks = [
                    Benchmarks.MEAN_INTER_TIME,
                    Benchmarks.MARK_DISTRIBUTION,
                    Benchmarks.INTERTIME_DISTRIBUTION,
                ]

                if run_all_configs and benchmark_manager.get_config_count() > 1:
                    results = benchmark_manager.run_multiple_on_all_configs(
                        default_benchmarks, **benchmark_params
                    )
                else:
                    results = benchmark_manager.run_multiple(
                        default_benchmarks, **benchmark_params
                    )

            # Afficher les résultats
            self._display_benchmark_results(results, output_dir)

            # Calculer le nombre de benchmarks exécutés
            if isinstance(results, dict):
                # Vérifier si c'est multi-config
                is_multi_config = any(isinstance(v, dict) for v in results.values())
                if is_multi_config:
                    num_benchmarks = sum(
                        len(dataset_results)
                        for dataset_results in results.values()
                        if isinstance(dataset_results, dict)
                    )
                else:
                    num_benchmarks = len(results)
            else:
                num_benchmarks = 1

            self.print_success(
                f"Benchmarks terminés - {num_benchmarks} benchmarks exécutés"
            )
            self.print_info(f"Résultats sauvegardés dans: {output_dir}")

            return True

        except Exception as e:
            self.print_error_with_traceback(f"Erreur lors du benchmark: {e}", e)
            if self.debug:
                self.logger.exception("Détails de l'erreur:")
            return False

    def _display_benchmark_results(self, results: Dict[str, Any], output_dir: str):
        """Affiche et sauvegarde les résultats des benchmarks."""

        descriptions = {
            "mean_inter_time": "Always predicts the mean of inter-event times",
            "lag1_mark_benchmark": "Predicts the last mark (previous event type)",
            "intertime_distribution_sampling": "Samples from empirical inter-time distribution",
            "mark_distribution_sampling": "Samples from empirical mark distribution",
        }

        # Vérifier si c'est un résultat multi-config (dict de dicts)
        # Multi-config a la structure: {dataset_id: {benchmark_name: result}}
        # Single-config a la structure: {benchmark_name: result}
        is_multi_config = False
        if results:
            first_value = next(iter(results.values()))
            # Si la première valeur est un dict avec plusieurs benchmarks
            is_multi_config = isinstance(first_value, dict) and any(
                key
                in [
                    "mean_inter_time",
                    "lag1_mark_benchmark",
                    "intertime_distribution_sampling",
                    "mark_distribution_sampling",
                ]
                for key in (first_value.keys() if isinstance(first_value, dict) else [])
            )

        if is_multi_config:
            # Affichage pour résultats multi-configs
            if self.console:
                from rich.table import Table

                table = Table(
                    title="Résultats des Benchmarks TPP - Multi-Configurations"
                )
                table.add_column("Dataset", style="magenta", no_wrap=True)
                table.add_column("Benchmark", style="cyan")
                table.add_column("Statut", style="green")
                table.add_column("Description", style="yellow")

                for dataset_id, dataset_results in results.items():
                    for benchmark_name, result in dataset_results.items():
                        status = "✓ Terminé" if result else "✗ Échec"
                        description = descriptions.get(benchmark_name, "TPP Benchmark")
                        table.add_row(dataset_id, benchmark_name, status, description)

                self.console.print(table)
            else:
                print("\n=== Résultats des Benchmarks - Multi-Configurations ===")
                for dataset_id, dataset_results in results.items():
                    print(f"\n--- Dataset: {dataset_id} ---")
                    for benchmark_name, result in dataset_results.items():
                        status = "✓ Terminé" if result else "✗ Échec"
                        print(f"  {status} {benchmark_name}")
        else:
            # Affichage pour résultats single-config
            if self.console:
                from rich.table import Table

                table = Table(title="Résultats des Benchmarks TPP")
                table.add_column("Benchmark", style="cyan")
                table.add_column("Statut", style="green")
                table.add_column("Description", style="yellow")

                for benchmark_name, result in results.items():
                    status = "✓ Terminé" if result else "✗ Échec"
                    description = descriptions.get(benchmark_name, "TPP Benchmark")
                    table.add_row(benchmark_name, status, description)

                self.console.print(table)
            else:
                print("\n=== Résultats des Benchmarks ===")
                for benchmark_name, result in results.items():
                    status = "✓ Terminé" if result else "✗ Échec"
                    print(f"{status} {benchmark_name}")

        # Sauvegarde des résultats
        self._save_benchmark_results(results, output_dir)

    def _save_benchmark_results(self, results: Dict[str, Any], output_dir: str):
        """Sauvegarde les résultats des benchmarks."""
        try:
            output_path = Path(output_dir)

            # Vérifier si c'est un résultat multi-config
            is_multi_config = any(isinstance(v, dict) for v in results.values())

            if is_multi_config:
                # Calcul des statistiques pour multi-configs
                total_benchmarks = sum(
                    len(dataset_results) for dataset_results in results.values()
                )
                total_success = sum(
                    sum(1 for r in dataset_results.values() if r)
                    for dataset_results in results.values()
                )
                total_failed = total_benchmarks - total_success

                results_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "multi_config": True,
                    "num_configs": len(results),
                    "datasets": list(results.keys()),
                    "total_benchmarks_executed": total_benchmarks,
                    "successful_benchmarks": total_success,
                    "failed_benchmarks": total_failed,
                    "results": results,
                }
            else:
                # Statistiques pour single-config
                results_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "multi_config": False,
                    "benchmarks_executed": len(results),
                    "successful_benchmarks": sum(1 for r in results.values() if r),
                    "failed_benchmarks": sum(1 for r in results.values() if not r),
                    "results": results,
                }

            # Sauvegarder en JSON
            json_file = output_path / "benchmark_results.json"
            with open(json_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            self.print_success(f"Résultats JSON sauvegardés: {json_file}")

            # Générer un rapport texte simple
            report_file = output_path / "benchmark_report.txt"
            with open(report_file, "w") as f:
                f.write(f"Rapport de Benchmark TPP\n")
                f.write(f"========================\n\n")
                f.write(f"Date: {results_data['timestamp']}\n")

                if is_multi_config:
                    f.write(f"Mode: Multi-Configurations\n")
                    f.write(
                        f"Nombre de configurations: {results_data['num_configs']}\n"
                    )
                    f.write(f"Datasets: {', '.join(results_data['datasets'])}\n")
                    f.write(
                        f"Total benchmarks exécutés: {results_data['total_benchmarks_executed']}\n"
                    )
                    f.write(f"Succès: {results_data['successful_benchmarks']}\n")
                    f.write(f"Échecs: {results_data['failed_benchmarks']}\n\n")

                    f.write("Détails des résultats par configuration:\n")
                    f.write("=" * 50 + "\n")
                    for dataset_id, dataset_results in results.items():
                        f.write(f"\nDataset: {dataset_id}\n")
                        f.write("-" * 30 + "\n")
                        for benchmark_name, result in dataset_results.items():
                            status = "SUCCÈS" if result else "ÉCHEC"
                            f.write(f"  {benchmark_name}: {status}\n")
                else:
                    f.write(f"Mode: Configuration unique\n")
                    f.write(
                        f"Benchmarks exécutés: {results_data['benchmarks_executed']}\n"
                    )
                    f.write(f"Succès: {results_data['successful_benchmarks']}\n")
                    f.write(f"Échecs: {results_data['failed_benchmarks']}\n\n")

                    f.write("Détails des résultats:\n")
                    f.write("-" * 30 + "\n")
                    for benchmark_name, result in results.items():
                        status = "SUCCÈS" if result else "ÉCHEC"
                        f.write(f"{benchmark_name}: {status}\n")

            self.print_success(f"Rapport texte sauvegardé: {report_file}")

        except Exception as e:
            self.print_error_with_traceback(f"Erreur lors de la sauvegarde: {e}", e)

    def list_available_benchmarks(self) -> List[str]:
        """Retourne la liste des benchmarks disponibles."""
        if Benchmarks is None:
            self.print_error("BenchmarksEnum non disponible")
            return []

        benchmarks = [benchmark.benchmark_name for benchmark in Benchmarks]

        if self.console:
            from rich.table import Table

            table = Table(title="Benchmarks TPP Disponibles")
            table.add_column("Nom", style="cyan")
            table.add_column("Description", style="yellow")

            descriptions = {
                "mean_inter_time": "Always predicts the mean of inter-event times",
                "lag1_mark_benchmark": "Predicts the last mark (previous event type)",
                "intertime_distribution_sampling": "Samples from empirical inter-time distribution",
                "mark_distribution_sampling": "Samples from empirical mark distribution",
            }

            for benchmark in Benchmarks:
                description = descriptions.get(
                    benchmark.benchmark_name, "TPP Benchmark"
                )
                table.add_row(benchmark.benchmark_name, description)

            self.console.print(table)
        else:
            print("\n=== Benchmarks Disponibles ===")
            for benchmark in benchmarks:
                print(f"- {benchmark}")

        return benchmarks
