"""
Registry simple pour les benchmarks

Enum des benchmarks disponibles pour faciliter l'utilisation.

Utilisation:
    from easy_tpp.evaluation.benchmarks.registry import Benchmarks

    # Lister tous les benchmarks
    print(list(Benchmarks))

    # Obtenir une classe de benchmark
    benchmark_class = Benchmarks.MEAN_INTER_TIME.get_class()
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Type, Union

from easy_tpp.evaluation.benchmarks.base_bench import BaseBenchmark
from easy_tpp.evaluation.benchmarks.last_mark_bench import LastMarkBenchmark
from easy_tpp.evaluation.benchmarks.mean_bench import MeanInterTimeBenchmark
from easy_tpp.evaluation.benchmarks.sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)
from easy_tpp.evaluation.benchmarks.sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


class Benchmarks(Enum):
    """Enum des benchmarks disponibles."""

    MEAN_INTER_TIME = ("mean_inter_time", MeanInterTimeBenchmark)
    LAG1_MARK = ("lag1_mark_benchmark", LastMarkBenchmark)
    INTERTIME_DISTRIBUTION = (
        "intertime_distribution_sampling",
        InterTimeDistributionBenchmark,
    )
    MARK_DISTRIBUTION = ("mark_distribution_sampling", MarkDistributionBenchmark)

    def __init__(self, benchmark_name: str, benchmark_class: Type[BaseBenchmark]):
        self.benchmark_name = benchmark_name
        self.benchmark_class = benchmark_class

    def get_class(self) -> Type[BaseBenchmark]:
        """Obtenir la classe du benchmark."""
        return self.benchmark_class

    def get_name(self) -> str:
        """Obtenir le nom du benchmark."""
        return self.benchmark_name

    @classmethod
    def get_benchmark_by_name(cls, name: str) -> "Benchmarks":
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
        data_config,
        dataset_name: str,
        save_dir: Optional[Union[Path, str]] = None,
    ):
        """
        Initialiser la factory.

        Args:
            data_config: Configuration des données
            dataset_name: Nom du dataset
            save_dir: Répertoire pour sauvegarder les résultats
        """
        self.data_config = data_config
        self.dataset_name = dataset_name
        self.save_dir = save_dir or ROOT_DIR / "artifacts" / dataset_name / "benchmarks"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def run_single(self, benchmark: Benchmarks, **kwargs):
        """Lancer un seul benchmark."""
        print(f"Lancement du benchmark: {benchmark.benchmark_name}")

        benchmark_class = benchmark.get_class()
        instance = benchmark_class(
            data_config=self.data_config,
            dataset_name=self.dataset_name,
            save_dir=self.save_dir,
            **kwargs,
        )

        results = instance.evaluate()
        print(f"Benchmark {benchmark.benchmark_name} terminé")
        return results

    def run_multiple(self, benchmarks: list[Benchmarks], **kwargs):
        """Lancer plusieurs benchmarks."""
        results = {}
        for benchmark in benchmarks:
            results[benchmark.benchmark_name] = self.run_single(benchmark, **kwargs)
        return results

    def run_all(self, **kwargs):
        """Lancer tous les benchmarks disponibles."""
        all_benchmarks = list(Benchmarks)
        return self.run_multiple(all_benchmarks, **kwargs)

    def run_by_names(self, benchmark_names: list[str], **kwargs):
        """Lancer des benchmarks par leurs noms."""
        benchmarks = []
        for name in benchmark_names:
            benchmark = Benchmarks.get_benchmark_by_name(name)
            benchmarks.append(benchmark)
        return self.run_multiple(benchmarks, **kwargs)
