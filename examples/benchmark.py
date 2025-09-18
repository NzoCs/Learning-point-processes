from easy_tpp.config_factory import DataConfig
from easy_tpp.evaluation.benchmarks.mean_bench import MeanInterTimeBenchmark
from easy_tpp.evaluation.benchmarks.sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)
from easy_tpp.evaluation.benchmarks.sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)


def benchmark_mean_intertime() -> None:
    """Benchmark mean inter-time prediction."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    benchmark = MeanInterTimeBenchmark(
        data_config=data_config,
        experiment_id="mean_test",
        save_dir="./benchmark_results",
    )

    results = benchmark.evaluate()
    print(f"Mean benchmark completed: {results}")


def benchmark_mark_distribution() -> None:
    """Benchmark mark distribution sampling."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    benchmark = MarkDistributionBenchmark(
        data_config=data_config,
        experiment_id="mark_test",
        save_dir="./benchmark_results",
    )

    results = benchmark.evaluate()
    print(f"Mark distribution benchmark completed: {results}")


def benchmark_intertime_distribution() -> None:
    """Benchmark inter-time distribution sampling."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    benchmark = InterTimeDistributionBenchmark(
        data_config=data_config,
        experiment_id="intertime_test",
        save_dir="./benchmark_results",
    )

    results = benchmark.evaluate()
    print(f"Inter-time distribution benchmark completed: {results}")


def benchmark_multiple_on_dataset() -> None:
    """Run multiple benchmarks on same dataset."""
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    benchmarks = [
        ("mean", MeanInterTimeBenchmark),
        ("mark_dist", MarkDistributionBenchmark),
        ("time_dist", InterTimeDistributionBenchmark),
    ]

    for name, benchmark_class in benchmarks:
        print(f"Running {name} benchmark...")
        benchmark = benchmark_class(
            data_config=data_config,
            experiment_id=f"{name}_test",
            save_dir="./benchmark_results",
        )
        results = benchmark.evaluate()
        print(f"{name} completed")


def main() -> None:
    benchmark_mean_intertime()
    benchmark_mark_distribution()
    benchmark_intertime_distribution()
    benchmark_multiple_on_dataset()


if __name__ == "__main__":
    main()
