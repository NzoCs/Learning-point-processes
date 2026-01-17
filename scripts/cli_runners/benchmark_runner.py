"""
Benchmark Runner

Runner pour les tests de performance et benchmarking TPP.
"""

from typing import List, Optional, Union

from new_ltpp.configs.config_loaders.data_config_loader import DataConfigYamlLoader
from new_ltpp.configs import DataConfigBuilder
from new_ltpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarkManager,
)
from new_ltpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarksEnum as Benchmarks,
)

from .cli_base import CLIRunnerBase


class BenchmarkRunner(CLIRunnerBase):
    """
    Runner pour les tests de performance et benchmarking.
    Mesure les temps d'exécution, utilisation mémoire, et performance des modèles.
    """

    def __init__(self, debug: bool = False):
        super().__init__("BenchmarkRunner", debug=debug)

    def run_benchmark(
        self,
        data_config: Optional[Union[str, List[str]]],
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
            run_all_configs: Exécuter sur toutes les configurations disponibles dans le YAML
            **benchmark_params: Paramètres supplémentaires pour les benchmarks

        Returns:
            True si les benchmarks se sont déroulés avec succès
        """
        # Vérifier les dépendances
        required_modules = ["new_ltpp.configs", "new_ltpp.evaluation.benchmarks"]
        if not self.check_dependencies(required_modules):
            return False

        # Configuration par défaut si aucun fichier spécifié
        if config_path is None:
            config_path = str(self.get_config_path())
            self.print_info(
                f"Utilisation de la configuration par défaut: {config_path}"
            )

        # Si run_all_configs, récupérer toutes les configurations du YAML
        if run_all_configs:
            import yaml

            self.print_info("Récupération de toutes les configurations disponibles...")

            with open(config_path, "r") as f:
                yaml_content = yaml.safe_load(f)

            # Extraire tous les noms de configs data disponibles
            available_data_configs = list(yaml_content.get("data_configs", {}).keys())

            if not available_data_configs:
                self.print_error("Aucune configuration de données trouvée dans le YAML")
                return False

            self.print_info(
                f"Configurations trouvées: {', '.join(available_data_configs)}"
            )
            data_configs_list = available_data_configs
        else:
            # Support de plusieurs configurations spécifiées
            data_configs_list = (
                data_config if isinstance(data_config, list) else [data_config]
            )

        # Construire toutes les configurations
        all_data_configs = []
        for data_cfg in data_configs_list:
            try:
                # Construire les chemins de configuration avec la méthode utilitaire
                config_paths = self._build_config_paths(
                    data=data_cfg, data_loading=data_loading_config
                )

                self.print_info(
                    f"Configuration des données: {config_paths.get('data_config_path')}"
                )

                # Use Loader to get dictionary
                loader = DataConfigYamlLoader()
                config_dict = loader.load(
                    yaml_path=config_path,
                    data_config_path=config_paths.get("data_config_path"),  # type: ignore
                    data_loading_config_path=config_paths.get(
                        "data_loading_config_path"
                    ),
                )

                # Use Builder to create object
                builder = DataConfigBuilder()
                builder.from_dict(config_dict)
                built_config = builder.build()
                all_data_configs.append(built_config)

                self.print_info(f"Configuration chargée: {built_config.dataset_id}")

            except Exception as e:
                self.print_error(f"Erreur lors du chargement de {data_cfg}: {e}")
                if self.debug:
                    self.logger.exception(f"Détails de l'erreur pour {data_cfg}:")
                # Continue avec les autres configs

        if not all_data_configs:
            self.print_error("Aucune configuration n'a pu être chargée")
            return False

        # Créer le BenchmarkManager
        benchmark_manager = BenchmarkManager(
            base_dir=output_dir or self.get_output_path()
        )

        try:
            self.print_info("Configuration du benchmark...")
            self.print_info(
                f"Exécution sur {len(all_data_configs)} configuration(s)..."
            )
            dataset_ids = [cfg.dataset_id for cfg in all_data_configs]
            self.print_info(f"Datasets: {', '.join(dataset_ids)}")

            # Déterminer quels benchmarks exécuter
            if run_all:
                self.print_info("Exécution de tous les benchmarks disponibles...")
                benchmark_manager.run_all_benchmarks(
                    all_data_configs, **benchmark_params
                )

            elif benchmarks:
                self.print_info(f"Exécution des benchmarks: {benchmarks}")
                benchmark_manager.run_by_names(
                    benchmarks, all_data_configs, **benchmark_params
                )

            else:
                # Par défaut, exécuter quelques benchmarks essentiels
                self.print_info("Exécution des benchmarks par défaut...")
                default_benchmarks = [
                    Benchmarks.MEAN_INTER_TIME,
                    Benchmarks.MARK_DISTRIBUTION,
                    Benchmarks.INTERTIME_DISTRIBUTION,
                ]
                benchmark_manager.run(
                    default_benchmarks, all_data_configs, **benchmark_params
                )

            self.print_info(f"Résultats sauvegardés dans: {benchmark_manager.base_dir}")

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
