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
    from easy_tpp.configs import ConfigFactory, ConfigType
    from easy_tpp.runners import RunnerManager
    import psutil
    import torch
except ImportError as e:
    ConfigFactory = None
    RunnerManager = None
    psutil = None
    torch = None
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
        configs: List[str],
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        output_dir: str = "./benchmarks",
        iterations: int = 3,
        include_memory: bool = True,
        include_gpu: bool = True
    ) -> bool:
        """
        Lance un benchmark de performance sur plusieurs configurations.
        
        Args:
            configs: Liste des fichiers de configuration à tester
            models: Liste des modèles à tester (optionnel)
            datasets: Liste des datasets à tester (optionnel)
            output_dir: Répertoire de sortie des résultats
            iterations: Nombre d'itérations par test
            include_memory: Inclure les mesures de mémoire
            include_gpu: Inclure les mesures GPU
            
        Returns:
            True si le benchmark s'est déroulé avec succès
        """
        # Vérifier les dépendances
        required_modules = ["easy_tpp.config_factory", "easy_tpp.runners"]
        if not self.check_dependencies(required_modules):
            return False
            
        try:
            self.print_info(f"Lancement du benchmark - {len(configs)} configs, {iterations} itérations")
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Résultats globaux
            benchmark_results = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_info": self._collect_system_info(),
                "benchmark_config": {
                    "iterations": iterations,
                    "include_memory": include_memory,
                    "include_gpu": include_gpu
                },
                "results": []
            }
            
            # Benchmark pour chaque configuration
            for config_path in configs:
                self.print_info(f"Test de configuration: {config_path}")
                
                config_results = self._benchmark_config(
                    config_path, iterations, include_memory, include_gpu
                )
                benchmark_results["results"].append(config_results)
                
                # Affichage intermédiaire
                self._display_config_results(config_results)
            
            # Sauvegarde des résultats
            results_file = output_dir / f"benchmark_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            
            # Génération du rapport
            report_file = output_dir / "benchmark_report.html"
            self._generate_html_report(benchmark_results, report_file)
            
            # Résumé final
            self._display_final_summary(benchmark_results)
            
            self.print_success(f"Benchmark terminé - Résultats: {results_file}")
            return True
            
        except Exception as e:
            self.print_error(f"Erreur lors du benchmark: {e}")
            self.logger.exception("Détails de l'erreur:")
            return False
    
    def _benchmark_config(
        self,
        config_path: str,
        iterations: int,
        include_memory: bool,
        include_gpu: bool
    ) -> Dict[str, Any]:
        """Benchmark d'une configuration spécifique."""
        config_path = Path(config_path)
        
        results = {
            "config_name": config_path.stem,
            "config_path": str(config_path),
            "iterations": iterations,
            "timings": [],
            "memory_usage": [],
            "gpu_usage": [],
            "errors": []
        }
        
        for i in range(iterations):
            self.print_info(f"  Itération {i+1}/{iterations}")
            
            try:
                # Mesures avant exécution
                start_memory = self._get_memory_usage() if include_memory else 0
                start_gpu = self._get_gpu_usage() if include_gpu and torch and torch.cuda.is_available() else {}
                
                # Chronométrage de l'exécution
                start_time = time.time()
                
                # Charger et exécuter la configuration
                config = ConfigFactory.from_yaml(str(config_path), ConfigType.RUNNER)
                runner = RunnerManager(config)
                
                # Exécution (mode test rapide pour benchmark)
                runner.run(phase="train", max_epochs=5)  # Réduire pour benchmark
                
                end_time = time.time()
                
                # Mesures après exécution
                end_memory = self._get_memory_usage() if include_memory else 0
                end_gpu = self._get_gpu_usage() if include_gpu and torch and torch.cuda.is_available() else {}
                
                # Stocker les résultats
                execution_time = end_time - start_time
                results["timings"].append(execution_time)
                
                if include_memory:
                    memory_delta = end_memory - start_memory
                    results["memory_usage"].append(memory_delta)
                
                if include_gpu:
                    results["gpu_usage"].append({
                        "start": start_gpu,
                        "end": end_gpu
                    })
                
                self.print_success(f"    Temps: {execution_time:.2f}s")
                
            except Exception as e:
                error_msg = str(e)
                results["errors"].append(error_msg)
                self.print_error(f"    Erreur: {error_msg}")
        
        # Calculer les statistiques
        if results["timings"]:
            results["stats"] = {
                "mean_time": sum(results["timings"]) / len(results["timings"]),
                "min_time": min(results["timings"]),
                "max_time": max(results["timings"]),
                "success_rate": (iterations - len(results["errors"])) / iterations
            }
        
        return results
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collecte les informations système pour le benchmark."""
        info = {
            "python_version": __import__("sys").version,
            "platform": __import__("platform").platform()
        }
        
        # Informations PyTorch
        if torch:
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
        
        # Informations système
        if psutil:
            info["cpu_count"] = psutil.cpu_count()
            info["memory_total"] = psutil.virtual_memory().total
        
        return info
    
    def _get_memory_usage(self) -> float:
        """Retourne l'utilisation mémoire actuelle en MB."""
        if psutil:
            return psutil.Process().memory_info().rss / 1024 / 1024
        return 0.0
    
    def _get_gpu_usage(self) -> Dict[str, Any]:
        """Retourne l'utilisation GPU actuelle."""
        if not torch or not torch.cuda.is_available():
            return {}
        
        try:
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                gpu_info[f"gpu_{i}"] = {
                    "memory_allocated": torch.cuda.memory_allocated(i) / 1024 / 1024,
                    "memory_reserved": torch.cuda.memory_reserved(i) / 1024 / 1024
                }
            return gpu_info
        except:
            return {}
    
    def _display_config_results(self, results: Dict[str, Any]):
        """Affiche les résultats d'une configuration."""
        if not self.console:
            print(f"  Résultats pour {results['config_name']}:")
            if "stats" in results:
                print(f"    Temps moyen: {results['stats']['mean_time']:.2f}s")
                print(f"    Taux de réussite: {results['stats']['success_rate']:.1%}")
            return
        
        from rich.table import Table
        
        table = Table(title=f"Résultats: {results['config_name']}")
        table.add_column("Métrique", style="cyan")
        table.add_column("Valeur", style="magenta")
        
        if "stats" in results:
            stats = results["stats"]
            table.add_row("Temps moyen", f"{stats['mean_time']:.2f}s")
            table.add_row("Temps min", f"{stats['min_time']:.2f}s")
            table.add_row("Temps max", f"{stats['max_time']:.2f}s")
            table.add_row("Taux de réussite", f"{stats['success_rate']:.1%}")
        
        if results["errors"]:
            table.add_row("Erreurs", f"{len(results['errors'])}")
        
        self.console.print(table)
    
    def _display_final_summary(self, benchmark_results: Dict[str, Any]):
        """Affiche le résumé final du benchmark."""
        if not self.console:
            print("\n=== Résumé du Benchmark ===")
            for result in benchmark_results["results"]:
                if "stats" in result:
                    print(f"{result['config_name']}: {result['stats']['mean_time']:.2f}s")
            return
        
        from rich.table import Table
        
        table = Table(title="Résumé du Benchmark")
        table.add_column("Configuration", style="cyan")
        table.add_column("Temps Moyen", style="magenta")
        table.add_column("Taux Réussite", style="green")
        table.add_column("Erreurs", style="red")
        
        for result in benchmark_results["results"]:
            if "stats" in result:
                table.add_row(
                    result["config_name"],
                    f"{result['stats']['mean_time']:.2f}s",
                    f"{result['stats']['success_rate']:.1%}",
                    str(len(result["errors"]))
                )
        
        self.console.print(table)
    
    def _generate_html_report(self, results: Dict[str, Any], output_file: Path):
        """Génère un rapport HTML détaillé."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TPP Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>TPP Benchmark Report</h1>
    <div class="summary">
        <h2>Informations Générales</h2>
        <p><strong>Date:</strong> {results['timestamp']}</p>
        <p><strong>Configurations testées:</strong> {len(results['results'])}</p>
        <p><strong>Itérations par config:</strong> {results['benchmark_config']['iterations']}</p>
    </div>
    
    <h2>Résultats Détaillés</h2>
    <table>
        <tr>
            <th>Configuration</th>
            <th>Temps Moyen (s)</th>
            <th>Temps Min (s)</th>
            <th>Temps Max (s)</th>
            <th>Taux Réussite</th>
            <th>Erreurs</th>
        </tr>
"""
        
        for result in results["results"]:
            if "stats" in result:
                html_content += f"""
        <tr>
            <td>{result['config_name']}</td>
            <td>{result['stats']['mean_time']:.2f}</td>
            <td>{result['stats']['min_time']:.2f}</td>
            <td>{result['stats']['max_time']:.2f}</td>
            <td>{result['stats']['success_rate']:.1%}</td>
            <td>{len(result['errors'])}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.print_success(f"Rapport HTML généré: {output_file}")