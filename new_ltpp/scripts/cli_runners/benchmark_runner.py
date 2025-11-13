"""
Benchmark Runner

Runner pour les tests de performance et benchmarking TPP.
"""

from typing import Any, Dict, List, Optional, Union

from .cli_base import CONFIG_MAP, CLIRunnerBase


from new_ltpp.configs import DataConfigBuilder
from new_ltpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarkManager,
)
from new_ltpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarksEnum as Benchmarks,
    )

class BenchmarkRunner(CLIRunnerBase):
    """
    Runner pour les tests de performance et benchmarking.
    Mesure les temps d'exécution, utilisation mémoire, et performance des modèles.
    """

    def __init__(self, debug: bool = False):
        super().__init__("BenchmarkRunner", debug=debug)

    def run_benchmark(
        self,
        data_config: Union[str, List[str]],
        data_loading_config: str,
        config_path: Optional[str] = None,
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
        
        # Support de plusieurs configurations
        data_configs_list = (
            data_config if isinstance(data_config, list) else [data_config]
        )

        
        # Configuration par défaut si aucun fichier spécifié
        if config_path is None:
            config_path = str(self.get_config_path())
            self.print_info(
                f"Utilisation de la configuration par défaut: {config_path}"
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

        # Créer le BenchmarkManager
        benchmark_manager = BenchmarkManager(save_dir=output_dir or self.get_output_path("benchmarks"))

        try:
            self.print_info("Configuration du benchmark...")

            # Déterminer quels benchmarks exécuter
            if run_all and run_all_configs and len(all_data_configs) > 1:
                self.print_info(
                    f"Exécution de tous les benchmarks sur {len(all_data_configs)} configurations..."
                )
                dataset_ids = [cfg.dataset_id for cfg in all_data_configs]
                self.print_info(
                    f"Datasets: {', '.join(dataset_ids)}"
                )
                results = benchmark_manager.run_all_benchmarks(all_data_configs, **benchmark_params)

            elif run_all:
                self.print_info("Exécution de tous les benchmarks disponibles...")
                results = benchmark_manager.run_all_benchmarks(all_data_configs, **benchmark_params)

            elif benchmarks:
                self.print_info(f"Exécution des benchmarks: {benchmarks}")
                results = benchmark_manager.run_by_names(benchmarks, all_data_configs, **benchmark_params)

            else:
                # Par défaut, exécuter quelques benchmarks essentiels
                self.print_info("Exécution des benchmarks par défaut...")
                default_benchmarks = [
                    Benchmarks.MEAN_INTER_TIME,
                    Benchmarks.MARK_DISTRIBUTION,
                    Benchmarks.INTERTIME_DISTRIBUTION,
                ]
                results = benchmark_manager.run(default_benchmarks, all_data_configs, **benchmark_params)

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
            self.print_info(f"Résultats sauvegardés dans: {benchmark_manager.save_dir}")

            return True

        except Exception as e:
            self.print_error_with_traceback(f"Erreur lors du benchmark: {e}", e)
            if self.debug:
                self.logger.exception("Détails de l'erreur:")
            return False

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
