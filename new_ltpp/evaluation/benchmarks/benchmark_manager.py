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
from typing import Dict, Optional, Type, Union, Any

from new_ltpp.configs import DataConfig
from new_ltpp.globals import OUTPUT_DIR


from .base_bench import BaseBenchmark
from .bench_interfaces import BenchmarkInterface
from .last_mark_bench import LastMarkBenchmark
from .mean_bench import MeanInterTimeBenchmark
from .sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)
from .sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)

class BenchmarksEnum(Enum):
    """Enum of available benchmarks."""

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
        """Return the benchmark class."""
        return self.benchmark_class

    def get_name(self) -> str:
        """Return the benchmark name."""
        return self.benchmark_name

    @classmethod
    def get_benchmark_by_name(cls, name: str) -> Type:
        """Get a benchmark by its name."""
        for benchmark in cls:
            if benchmark.get_name() == name:
                return benchmark.get_class()
        raise ValueError(
            f"Benchmark '{name}' not found. Available: {[b.get_name() for b in cls]}"
        )

    @classmethod
    def list_names(cls) -> list[str]:
        """List all benchmark names."""
        return [benchmark.get_name() for benchmark in cls]


class BenchmarkManager:
    """Manager to run benchmarks on data configurations."""

    def __init__(self, save_dir: Union[Path, str] = OUTPUT_DIR / "benchmarks"):
        """
        Initialize the manager.

        Args:
            save_dir: Directory to save results (default: OUTPUT_DIR/benchmarks)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        benchmarks: Union[BenchmarksEnum, list[BenchmarksEnum]],
        data_configs: Union[DataConfig, list[DataConfig]],
        save_dir: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run benchmarks on data configurations.

        Args:
            benchmarks: Benchmark(s) to run
            data_configs: Data configuration(s) to use
            save_dir: Override the default save directory (optional)
            **kwargs: Additional arguments for the benchmarks

        Returns:
            Dict with results. Structure depends on input:
            - Single config, single benchmark: {benchmark_name: result}
            - Single config, multiple benchmarks: {benchmark_name: result, ...}
            - Multiple configs, any benchmarks: {dataset_id: {benchmark_name: result, ...}, ...}
        """
        # Normalize inputs to lists
        if isinstance(benchmarks, BenchmarksEnum):
            benchmarks = [benchmarks]
        if not isinstance(data_configs, list):
            data_configs = [data_configs]

        # Determine save directory
        output_dir = Path(save_dir) if save_dir else self.save_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Multiple configs: return nested dict {dataset_id: {benchmark_name: result}}
        if len(data_configs) > 1:
            results = {}
            for config in data_configs:
                dataset_id = config.dataset_id
                dataset_results = {}

                for bm in benchmarks:
                    benchmark_class = bm.get_class()
                    instance = benchmark_class(
                        data_config=config,
                        save_dir=output_dir / dataset_id,
                        **kwargs,
                    )
                    dataset_results[bm.benchmark_name] = instance.evaluate()
                    print(f"Benchmark {bm.benchmark_name} finished for {dataset_id}")

                results[dataset_id] = dataset_results

            return results

        # Single config: return flat dict {benchmark_name: result}
        config = data_configs[0]
        dataset_id = config.dataset_id
        results = {}

        for bm in benchmarks:
            benchmark_class = bm.get_class()
            instance = benchmark_class(
                data_config=config,
                save_dir=output_dir,
                **kwargs,
            )
            results[bm.benchmark_name] = instance.evaluate()
            print(f"Benchmark {bm.benchmark_name} finished for {dataset_id}")

        return results

    def run_all_benchmarks(
        self,
        data_configs: Union[DataConfig, list[DataConfig]],
        save_dir: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run all available benchmarks on data configurations.

        Args:
            data_configs: Data configuration(s) to use
            save_dir: Override the default save directory (optional)
            **kwargs: Additional arguments for the benchmarks

        Returns:
            Dict with results (see run() for structure details)
        """
        all_benchmarks = list(BenchmarksEnum)
        return self.run(all_benchmarks, data_configs, save_dir, **kwargs)

    def run_by_names(
        self,
        benchmark_names: list[str],
        data_configs: Union[DataConfig, list[DataConfig]],
        save_dir: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run benchmarks by their names on data configurations.

        Args:
            benchmark_names: List of benchmark names to run
            data_configs: Data configuration(s) to use
            save_dir: Override the default save directory (optional)
            **kwargs: Additional arguments for the benchmarks

        Returns:
            Dict with results (see run() for structure details)
        """
        benchmarks = []
        for name in benchmark_names:
            try:
                benchmark_enum = next(
                    b for b in BenchmarksEnum if b.get_name() == name
                )
                benchmarks.append(benchmark_enum)
            except StopIteration:
                raise ValueError(
                    f"Benchmark '{name}' not found. Available: {BenchmarksEnum.list_names()}"
                )

        return self.run(benchmarks, data_configs, save_dir, **kwargs)
