"""
Example usage of the BenchmarkManager with the enum

This example shows how to use the manager to simplify
running benchmarks compared to the previous code.
"""

from new_ltpp.configs import DataConfigBuilder
from new_ltpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarkManager,
)
from new_ltpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarksEnum as Benchmarks,
)
from new_ltpp.globals import CONFIGS_FILE

def example_simple_benchmark():
    """Simple example with a single benchmark."""
    # Use DataConfigBuilder to construct the config
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path=CONFIGS_FILE,
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    data_config = builder.build()

    manager = BenchmarkManager()
    results = manager.run(Benchmarks.MEAN_INTER_TIME, data_config)
    print(f"Results: {results}")


def example_multiple_benchmarks():
    """Example with multiple benchmarks."""
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path=CONFIGS_FILE,
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )

    data_config = builder.build()

    manager = BenchmarkManager()

    selected_benchmarks = [
        Benchmarks.MEAN_INTER_TIME,
        Benchmarks.MARK_DISTRIBUTION,
        Benchmarks.INTERTIME_DISTRIBUTION,
    ]

    results = manager.run(selected_benchmarks, data_config)

    for benchmark_name, result in results.items():
        print(f"Results for {benchmark_name}: finished")


def example_all_benchmarks():
    """Example to run all benchmarks."""
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path=CONFIGS_FILE,
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    data_config = builder.build()

    manager = BenchmarkManager()
    results = manager.run_all_benchmarks(data_config)
    print(f"All benchmarks completed: {len(results)} benchmarks")


def example_by_names():
    """Example to run benchmarks by their names."""
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path=CONFIGS_FILE,
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    data_config = builder.build()

    manager = BenchmarkManager()

    benchmark_names = ["mean_inter_time", "mark_distribution_sampling"]
    results = manager.run_by_names(benchmark_names, data_config)

    print(f"Benchmarks launched: {list(results.keys())}")


def example_with_parameters():
    """Example with custom parameters."""
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path=CONFIGS_FILE,
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    data_config = builder.build()

    manager = BenchmarkManager()

    results = manager.run(
        Benchmarks.INTERTIME_DISTRIBUTION,
        data_config,
        num_bins=100,  # Custom parameter for this benchmark
    )

    print("Benchmark with custom parameters finished")


def example_multiple_configs():
    """Example to run benchmarks on multiple data configurations."""
    builder1 = DataConfigBuilder()
    builder1.load_from_yaml(
        yaml_path=CONFIGS_FILE,
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    config1 = builder1.build()

    builder2 = DataConfigBuilder()
    builder2.load_from_yaml(
        yaml_path=CONFIGS_FILE,
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    config2 = builder2.build()

    manager = BenchmarkManager()

    # Run on multiple configs returns nested dict: {dataset_id: {benchmark_name: result}}
    results = manager.run_all_benchmarks([config1, config2])
    
    print(f"Benchmarks completed on {len(results)} datasets:")
    for dataset_id in results.keys():
        print(f"  - {dataset_id}: {len(results[dataset_id])} benchmarks")


def main():
    """Main function to run all examples."""
    print("=== Simple example ===")
    example_simple_benchmark()

    print("\n=== Multiple benchmarks example ===")
    example_multiple_benchmarks()

    print("\n=== All benchmarks example ===")
    example_all_benchmarks()

    print("\n=== By names example ===")
    example_by_names()

    print("\n=== With parameters example ===")
    example_with_parameters()

    print("\n=== Multiple configs example ===")
    example_multiple_configs()

    print("\n=== Available benchmarks list ===")
    print("Available benchmarks:")
    for benchmark in Benchmarks:
        print(f"- {benchmark.benchmark_name}")


if __name__ == "__main__":
    main()
