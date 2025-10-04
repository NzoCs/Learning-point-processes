"""
Example usage of the BenchmarkFactory with the enum

This example shows how to use the factory to simplify
running benchmarks compared to the previous code.
"""

from new_ltpp.configs import DataConfigBuilder
from new_ltpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarkManager,
)
from new_ltpp.evaluation.benchmarks.benchmark_manager import (
    BenchmarksEnum as Benchmarks,
)


def example_simple_benchmark():
    """Simple example with a single benchmark."""
    # Use DataConfigBuilder to construct the config
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path="../yaml_configs/configs.yaml",  # adapt according to your YAML file
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    data_config = builder.build()

    factory = BenchmarkManager(data_config)
    results = factory.run_single(Benchmarks.MEAN_INTER_TIME)
    print(f"Results: {results}")


def example_multiple_benchmarks():
    """Example with multiple benchmarks."""
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path="../yaml_configs/configs.yaml",  # adapt according to your YAML file
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )

    data_config = builder.build()

    factory = BenchmarkManager(data_config)

    selected_benchmarks = [
        Benchmarks.MEAN_INTER_TIME,
        Benchmarks.MARK_DISTRIBUTION,
        Benchmarks.INTERTIME_DISTRIBUTION,
    ]

    results = factory.run_multiple(selected_benchmarks)

    for benchmark_name, result in results.items():
        print(f"Results for {benchmark_name}: finished")


def example_all_benchmarks():
    """Example to run all benchmarks."""
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path="../yaml_configs/configs.yaml",  # adapt according to your YAML file
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    data_config = builder.build()

    factory = BenchmarkManager(data_config)
    results = factory.run_all()
    print(f"All benchmarks completed: {len(results)} benchmarks")


def example_by_names():
    """Example to run benchmarks by their names."""
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path="../yaml_configs/configs.yaml",  # adapt according to your YAML file
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    data_config = builder.build()

    factory = BenchmarkManager(data_config)

    benchmark_names = ["mean_inter_time", "mark_distribution_sampling"]
    results = factory.run_by_names(benchmark_names)

    print(f"Benchmarks launched: {list(results.keys())}")


def example_with_parameters():
    """Example with custom parameters."""
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path="../yaml_configs/configs.yaml",  # à adapter selon votre fichier YAML
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
    )
    data_config = builder.build()

    factory = BenchmarkManager(data_config)

    results = factory.run_single(
        Benchmarks.INTERTIME_DISTRIBUTION,
        num_bins=100,  # Custom parameter for this benchmark
    )

    print("Benchmark with custom parameters finished")


def main():
    """Main function to run all examples."""
    print("=== Simple example ===")
    example_simple_benchmark()

    print("\n=== Multiple example ===")
    example_multiple_benchmarks()

    print("\n=== All benchmarks example ===")
    example_all_benchmarks()

    print("\n=== By names example ===")
    example_by_names()

    print("\n=== With parameters example ===")
    example_with_parameters()

    print("\n=== Available benchmarks list ===")
    print("Available benchmarks:")
    for benchmark in Benchmarks:
        print(f"- {benchmark.benchmark_name}")


if __name__ == "__main__":
    main()
