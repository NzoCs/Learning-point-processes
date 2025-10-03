"""
Simple registry for benchmarks

Enum of available benchmarks to make usage easier.

Usage:
    from new_ltpp.evaluation.benchmarks.registry import Benchmarks

    # List all benchmarks
    print(list(Benchmarks))

    # Get a benchmark class
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
    """Enum of available benchmarks."""

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
        """Return the benchmark class."""
        return self.benchmark_class

    def get_name(self) -> str:
        """Return the benchmark name."""
        return self.benchmark_name

    @classmethod
    def get_benchmark_by_name(cls, name: str) -> BenchmarkInterface:
        """Get a benchmark by its name."""
        for benchmark in cls:
            if benchmark.benchmark_name == name:
                return benchmark
        raise ValueError(
            f"Benchmark '{name}' not found. Available: {[b.benchmark_name for b in cls]}"
        )

    @classmethod
    def list_names(cls) -> list[str]:
        """List all benchmark names."""
        return [benchmark.benchmark_name for benchmark in cls]


class BenchmarkManager:
    """Simple factory to run benchmarks using the enum."""

    def __init__(
        self,
        data_config: Union[DataConfig, list[DataConfig]],
        save_dir: Optional[Union[Path, str]] = None,
    ):
        """
        Initialize the factory.

        Args:
            data_config: Data configuration or list of configurations
            save_dir: Directory to save results
        """
        # Support a single config or a list of configs
        if isinstance(data_config, list):
            self.data_configs = data_config
            self.data_config = data_config[0]  # Default config
            self.dataset_name = data_config[0].dataset_id
        else:
            self.data_configs = [data_config]
            self.data_config = data_config
            self.dataset_name = data_config.dataset_id
        
        self.save_dir = save_dir or ROOT_DIR / "benchmarks"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def run_single(self, benchmark: BenchmarksEnum, data_config: Optional[DataConfig] = None, **kwargs):
        """
        Run a single benchmark on a configuration.

        Args:
            benchmark: Benchmark to run
            data_config: Specific configuration (uses self.data_config if None)
            **kwargs: Additional arguments for the benchmark
        """
        config = data_config or self.data_config
        dataset_name = config.dataset_id

        print(f"Running benchmark: {benchmark.benchmark_name} on {dataset_name}")

        benchmark_class = benchmark.get_class()
        instance = benchmark_class(
            data_config=config,
            dataset_name=dataset_name,
            save_dir=self.save_dir,
            **kwargs,
        )

        results = instance.evaluate()
        print(f"Benchmark {benchmark.benchmark_name} finished for {dataset_name}")
        return results

    def run_multiple(self, benchmarks: list[BenchmarksEnum], **kwargs):
        """Run multiple benchmarks."""
        results = {}
        for benchmark in benchmarks:
            results[benchmark.benchmark_name] = self.run_single(benchmark, **kwargs)
        return results

    def run_all(self, **kwargs):
        """Run all available benchmarks."""
        all_benchmarks = list(BenchmarksEnum)
        return self.run_multiple(all_benchmarks, **kwargs)

    def run_by_names(self, benchmark_names: list[str], **kwargs):
        """Run benchmarks by their names."""
        benchmarks = []
        for name in benchmark_names:
            benchmark = BenchmarksEnum.get_benchmark_by_name(name)
            benchmarks.append(benchmark)
        return self.run_multiple(benchmarks, **kwargs)
    
    def run_single_on_all_configs(self, benchmark: BenchmarksEnum, **kwargs):
        """
        Run a single benchmark on all configurations.

        Args:
            benchmark: Benchmark to run
            **kwargs: Additional arguments for the benchmark

        Returns:
            Dict[str, Any]: Results keyed by dataset_id
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
        Run multiple benchmarks on all configurations.

        Args:
            benchmarks: List of benchmarks to run
            **kwargs: Additional arguments for the benchmarks

        Returns:
            Dict[str, Dict[str, Any]]: Results by dataset_id then by benchmark
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
        Run all benchmarks on all configurations.

        Args:
            **kwargs: Additional arguments for the benchmarks

        Returns:
            Dict[str, Dict[str, Any]]: Results by dataset_id then by benchmark
        """
        all_benchmarks = list(BenchmarksEnum)
        return self.run_multiple_on_all_configs(all_benchmarks, **kwargs)
    
    def get_config_count(self) -> int:
        """Return the number of configurations."""
        return len(self.data_configs)
    
    def list_datasets(self) -> list[str]:
        """List all available dataset_ids."""
        return [config.dataset_id for config in self.data_configs]
