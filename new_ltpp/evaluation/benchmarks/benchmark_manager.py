"""
Registry simple pour les benchmarks

Enum des benchmarks disponibles pour faciliter l'utilisation.

Utilisation:
    from new_ltpp.evaluation.benchmarks.registry import Benchmarks

    # Lister tous les benchmarks
    print(list(Benchmarks))

    # Obtenir une classe de benchmark
    benchmark_class = Benchmarks.MEAN_INTER_TIME.get_class()
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Type, Union

from .base_bench import Benchmark
from .bench_interfaces import BenchmarkInterface
from .last_mark_bench import LastMarkBenchmark
from .mean_bench import MeanInterTimeBenchmark
from .sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)
from .sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)

from new_ltpp.configs import DataConfig

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts"

class BenchmarksEnum(Enum):
    """Enum des benchmarks disponibles."""

    MEAN_INTER_TIME = ("mean_inter_time", MeanInterTimeBenchmark)
    LAG1_MARK = ("lag1_mark_benchmark", LastMarkBenchmark)
    INTERTIME_DISTRIBUTION = (
        "intertime_distribution_sampling",
        InterTimeDistributionBenchmark,
    )
    MARK_DISTRIBUTION = ("mark_distribution_sampling", MarkDistributionBenchmark)

    def __init__(self, benchmark_name: str, benchmark_class: Type[Benchmark]):
        self.benchmark_name = benchmark_name
        self.benchmark_class = benchmark_class

    def get_class(self) -> Type[Benchmark]:
        """Obtenir la classe du benchmark."""
        return self.benchmark_class

    def get_name(self) -> str:
        """Obtenir le nom du benchmark."""
        return self.benchmark_name

    @classmethod
    def get_benchmark_by_name(cls, name: str) -> BenchmarkInterface:
        """Obtenir un benchmark par son nom."""
        for benchmark in cls:
            if benchmark.benchmark_name == name:
                return benchmark
        raise ValueError(
            f"Benchmark '{name}' introuvable. Disponibles: {[b.benchmark_name for b in cls]}"
        )

    @classmethod
    def list_names(cls) -> list[str]:
        """Lister tous les noms des benchmarks."""
        return [benchmark.benchmark_name for benchmark in cls]


class BenchmarkManager:
    """Factory simple pour lancer les benchmarks en utilisant l'enum."""

    def __init__(
        self,
        data_config: Union[DataConfig, list[DataConfig]],
        save_dir: Optional[Union[Path, str]] = None,
    ):
        """
        Initialiser la factory.

        Args:
            data_config: Configuration des données ou liste de configurations
            save_dir: Répertoire pour sauvegarder les résultats
        """
        # Support d'une seule config ou d'une liste de configs
        if isinstance(data_config, list):
            self.data_configs = data_config
            self.data_config = data_config[0]  # Config par défaut
            self.dataset_name = data_config[0].dataset_id
        else:
            self.data_configs = [data_config]
            self.data_config = data_config
            self.dataset_name = data_config.dataset_id
        
        self.save_dir = save_dir or ROOT_DIR / "benchmarks"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def run_single(self, benchmark: BenchmarksEnum, data_config: Optional[DataConfig] = None, **kwargs):
        """
        Lancer un seul benchmark sur une configuration.
        
        Args:
            benchmark: Benchmark à lancer
            data_config: Configuration spécifique (utilise self.data_config si None)
            **kwargs: Arguments supplémentaires pour le benchmark
        """
        config = data_config or self.data_config
        dataset_name = config.dataset_id
        
        print(f"Lancement du benchmark: {benchmark.benchmark_name} sur {dataset_name}")

        benchmark_class = benchmark.get_class()
        instance = benchmark_class(
            data_config=config,
            dataset_name=dataset_name,
            save_dir=self.save_dir,
            **kwargs,
        )

        results = instance.evaluate()
        print(f"Benchmark {benchmark.benchmark_name} terminé pour {dataset_name}")
        return results

    def run_multiple(self, benchmarks: list[BenchmarksEnum], **kwargs):
        """Lancer plusieurs benchmarks."""
        results = {}
        for benchmark in benchmarks:
            results[benchmark.benchmark_name] = self.run_single(benchmark, **kwargs)
        return results

    def run_all(self, **kwargs):
        """Lancer tous les benchmarks disponibles."""
        all_benchmarks = list(BenchmarksEnum)
        return self.run_multiple(all_benchmarks, **kwargs)

    def run_by_names(self, benchmark_names: list[str], **kwargs):
        """Lancer des benchmarks par leurs noms."""
        benchmarks = []
        for name in benchmark_names:
            benchmark = BenchmarksEnum.get_benchmark_by_name(name)
            benchmarks.append(benchmark)
        return self.run_multiple(benchmarks, **kwargs)
    
    def run_single_on_all_configs(self, benchmark: BenchmarksEnum, **kwargs):
        """
        Lancer un seul benchmark sur toutes les configurations.
        
        Args:
            benchmark: Benchmark à lancer
            **kwargs: Arguments supplémentaires pour le benchmark
            
        Returns:
            Dict[str, Any]: Résultats par dataset_id
        """
        results = {}
        for config in self.data_configs:
            dataset_id = config.dataset_id
            print(f"\n{'='*60}")
            print(f"Configuration: {dataset_id}")
            print(f"{'='*60}")
            results[dataset_id] = self.run_single(benchmark, data_config=config, **kwargs)
        return results
    
    def run_multiple_on_all_configs(self, benchmarks: list[BenchmarksEnum], **kwargs):
        """
        Lancer plusieurs benchmarks sur toutes les configurations.
        
        Args:
            benchmarks: Liste de benchmarks à lancer
            **kwargs: Arguments supplémentaires pour les benchmarks
            
        Returns:
            Dict[str, Dict[str, Any]]: Résultats par dataset_id puis par benchmark
        """
        results = {}
        for config in self.data_configs:
            dataset_id = config.dataset_id
            print(f"\n{'='*60}")
            print(f"Configuration: {dataset_id}")
            print(f"{'='*60}")
            results[dataset_id] = {}
            for benchmark in benchmarks:
                results[dataset_id][benchmark.benchmark_name] = self.run_single(
                    benchmark, data_config=config, **kwargs
                )
        return results
    
    def run_all_on_all_configs(self, **kwargs):
        """
        Lancer tous les benchmarks sur toutes les configurations.
        
        Args:
            **kwargs: Arguments supplémentaires pour les benchmarks
            
        Returns:
            Dict[str, Dict[str, Any]]: Résultats par dataset_id puis par benchmark
        """
        all_benchmarks = list(BenchmarksEnum)
        return self.run_multiple_on_all_configs(all_benchmarks, **kwargs)
    
    def get_config_count(self) -> int:
        """Retourne le nombre de configurations."""
        return len(self.data_configs)
    
    def list_datasets(self) -> list[str]:
        """Liste tous les dataset_ids disponibles."""
        return [config.dataset_id for config in self.data_configs]
