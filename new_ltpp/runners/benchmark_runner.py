"""
Benchmark Runner

Runner pour les tests de performance et benchmarking TPP.
"""

import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

from .cli_base import CLIRunnerBase

try:
    from new_ltpp.configs import DataConfigBuilder
    from new_ltpp.evaluation.benchmarks.benchmark_manager import (
        BenchmarkManager,
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
    
    def __init__(self):
        super().__init__("BenchmarkRunner")
        
    def run_benchmark(
        self,
        config_path: str = "yaml_configs/configs.yaml",
        data_config: str = "data_configs.test",
        data_loading_config: str = "data_loading_configs.quick_test",
        benchmarks: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        run_all: bool = False,
        **benchmark_params
    ) -> bool:
        """
        Lance des benchmarks TPP en utilisant le BenchmarkManager.
        
        Args:
            config_path: Chemin vers le fichier YAML de configuration
            data_config: Configuration des données (ex: 'data_configs.test')
            data_loading_config: Configuration du chargement des données
            benchmarks: Liste des noms de benchmarks à exécuter
            output_dir: Répertoire de sortie
            run_all: Exécuter tous les benchmarks disponibles
            **benchmark_params: Paramètres supplémentaires pour les benchmarks
            
        Returns:
            True si les benchmarks se sont déroulés avec succès
        """
        # Vérifier les dépendances
        required_modules = ["easy_tpp.configs", "easy_tpp.evaluation.benchmarks"]
        if not self.check_dependencies(required_modules):
            return False
            
        try:
            self.print_info("Configuration du benchmark...")
            
            # Créer le répertoire de sortie si nécessaire
            if output_dir is None:
                output_dir = str(self.get_output_path("benchmarks", f"benchmark_{int(time.time())}"))
            else:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Construire la configuration des données
            builder = DataConfigBuilder()
            builder.load_from_yaml(
                yaml_path=config_path,
                data_config_path=data_config,
                data_loading_config_path=data_loading_config
            )
            data_config = builder.build()
            
            self.print_info(f"Configuration chargée: {data_config}")
            
            # Créer le BenchmarkManager
            benchmark_manager = BenchmarkManager(data_config)
            
            # Déterminer quels benchmarks exécuter
            if run_all:
                self.print_info("Exécution de tous les benchmarks disponibles...")
                results = benchmark_manager.run_all(**benchmark_params)
                
            elif benchmarks:
                self.print_info(f"Exécution des benchmarks: {benchmarks}")
                
                # Convertir les noms en enum si nécessaire
                benchmark_enums = []
                for bench_name in benchmarks:
                    try:
                        # Essayer de trouver l'enum correspondant
                        benchmark_enum = next(
                            b for b in Benchmarks if b.benchmark_name.lower() == bench_name.lower()
                        )
                        benchmark_enums.append(benchmark_enum)
                    except StopIteration:
                        self.print_error(f"Benchmark non trouvé: {bench_name}")
                        continue
                
                if benchmark_enums:
                    results = benchmark_manager.run_multiple(benchmark_enums, **benchmark_params)
                else:
                    self.print_error("Aucun benchmark valide spécifié")
                    return False
                    
            else:
                # Par défaut, exécuter quelques benchmarks essentiels
                self.print_info("Exécution des benchmarks par défaut...")
                default_benchmarks = [
                    Benchmarks.MEAN_INTER_TIME,
                    Benchmarks.MARK_DISTRIBUTION,
                    Benchmarks.INTERTIME_DISTRIBUTION
                ]
                results = benchmark_manager.run_multiple(default_benchmarks, **benchmark_params)
            
            # Afficher les résultats
            self._display_benchmark_results(results, output_dir)
            
            self.print_success(f"Benchmarks terminés - {len(results)} benchmarks exécutés")
            self.print_info(f"Résultats sauvegardés dans: {output_dir}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Erreur lors du benchmark: {e}")
            self.logger.exception("Détails de l'erreur:")
            return False
    
    def _display_benchmark_results(self, results: Dict[str, Any], output_dir: str):
        """Affiche et sauvegarde les résultats des benchmarks."""
        
        # Affichage dans la console
        if self.console:
            from rich.table import Table
            
            table = Table(title="Résultats des Benchmarks TPP")
            table.add_column("Benchmark", style="cyan")
            table.add_column("Statut", style="green")
            table.add_column("Description", style="yellow")
            
            for benchmark_name, result in results.items():
                status = "✓ Terminé" if result else "✗ Échec"
                # Get benchmark description
                descriptions = {
                    'mean_inter_time': 'Always predicts the mean of inter-event times',
                    'lag1_mark_benchmark': 'Predicts the last mark (previous event type)',
                    'intertime_distribution_sampling': 'Samples from empirical inter-time distribution',
                    'mark_distribution_sampling': 'Samples from empirical mark distribution'
                }
                description = descriptions.get(benchmark_name, 'TPP Benchmark')
                
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
            
            # Sauvegarder en JSON
            results_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "benchmarks_executed": len(results),
                "successful_benchmarks": sum(1 for r in results.values() if r),
                "failed_benchmarks": sum(1 for r in results.values() if not r),
                "results": results
            }
            
            json_file = output_path / "benchmark_results.json"
            with open(json_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.print_success(f"Résultats JSON sauvegardés: {json_file}")
            
            # Générer un rapport texte simple
            report_file = output_path / "benchmark_report.txt"
            with open(report_file, 'w') as f:
                f.write(f"Rapport de Benchmark TPP\n")
                f.write(f"========================\n\n")
                f.write(f"Date: {results_data['timestamp']}\n")
                f.write(f"Benchmarks exécutés: {results_data['benchmarks_executed']}\n")
                f.write(f"Succès: {results_data['successful_benchmarks']}\n")
                f.write(f"Échecs: {results_data['failed_benchmarks']}\n\n")
                
                f.write("Détails des résultats:\n")
                f.write("-" * 30 + "\n")
                for benchmark_name, result in results.items():
                    status = "SUCCÈS" if result else "ÉCHEC"
                    f.write(f"{benchmark_name}: {status}\n")
            
            self.print_success(f"Rapport texte sauvegardé: {report_file}")
            
        except Exception as e:
            self.print_error(f"Erreur lors de la sauvegarde: {e}")
    
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
                'mean_inter_time': 'Always predicts the mean of inter-event times',
                'lag1_mark_benchmark': 'Predicts the last mark (previous event type)',
                'intertime_distribution_sampling': 'Samples from empirical inter-time distribution',
                'mark_distribution_sampling': 'Samples from empirical mark distribution'
            }
            
            for benchmark in Benchmarks:
                description = descriptions.get(benchmark.benchmark_name, 'TPP Benchmark')
                table.add_row(benchmark.benchmark_name, description)
            
            self.console.print(table)
        else:
            print("\n=== Benchmarks Disponibles ===")
            for benchmark in benchmarks:
                print(f"- {benchmark}")
        
        return benchmarks