"""
Benchmark Runner

Runner for performance tests and benchmarking of TPP.
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
    Runner for performance testing and benchmarking.
    Measures execution time, memory usage, and model performance.
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
        Run TPP benchmarks using the BenchmarkManager.

        Args:
            config_path: Path to the YAML configuration file
            data_config: Data configuration(s) (e.g. 'test' or ['test','large'])
            data_loading_config: Data loading configuration
            benchmarks: List of benchmark names to run
            output_dir: Output directory
            run_all: Run all available benchmarks
            run_all_configs: Run on all configurations available in the YAML
            **benchmark_params: Additional benchmark parameters

        Returns:
            True if benchmarks completed successfully
        """
        # Check dependencies
        required_modules = ["new_ltpp.configs", "new_ltpp.evaluation.benchmarks"]
        if not self.check_dependencies(required_modules):
            return False

        # Default configuration file if none provided
        if config_path is None:
            config_path = str(self.get_config_path())
            self.print_info(
                f"Using default configuration: {config_path}"
            )

        # If run_all_configs, retrieve all configurations from the YAML
        if run_all_configs:
            import yaml

            self.print_info("Gathering all available configurations...")

            with open(config_path, "r") as f:
                yaml_content = yaml.safe_load(f)

            # Extract all available data config names
            available_data_configs = list(yaml_content.get("data_configs", {}).keys())

            if not available_data_configs:
                self.print_error("No data configurations found in YAML")
                return False

            self.print_info(
                f"Found configurations: {', '.join(available_data_configs)}"
            )
            data_configs_list = available_data_configs
        else:
            # Support multiple specified configurations
            data_configs_list = (
                data_config if isinstance(data_config, list) else [data_config]
            )

        # Build all configurations
        all_data_configs = []
        for data_cfg in data_configs_list:
            try:
                # Build configuration paths with the helper method
                config_paths = self._build_config_paths(
                    data=data_cfg, data_loading=data_loading_config
                )

                self.print_info(
                    f"Data configuration path: {config_paths.get('data_config_path')}"
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

                self.print_info(f"Configuration loaded: {built_config.dataset_id}")

            except Exception as e:
                self.print_error(f"Error loading {data_cfg}: {e}")
                if self.debug:
                    self.logger.exception(f"Error details for {data_cfg}:")
                # Continue with other configs

        if not all_data_configs:
            self.print_error("No configuration could be loaded")
            return False

        # Create the BenchmarkManager
        benchmark_manager = BenchmarkManager(
            base_dir=output_dir or self.get_output_path()
        )

        try:
            self.print_info("Benchmark setup...")
            self.print_info(
                f"Running on {len(all_data_configs)} configuration(s)..."
            )
            dataset_ids = [cfg.dataset_id for cfg in all_data_configs]
            self.print_info(f"Datasets: {', '.join(dataset_ids)}")

            # Determine which benchmarks to run
            if run_all:
                self.print_info("Running all available benchmarks...")
                benchmark_manager.run_all_benchmarks(
                    all_data_configs, **benchmark_params
                )

            elif benchmarks:
                self.print_info(f"Running benchmarks: {benchmarks}")
                benchmark_manager.run_by_names(
                    benchmarks, all_data_configs, **benchmark_params
                )

            else:
                # By default, run a few essential benchmarks
                self.print_info("Running default benchmarks...")
                default_benchmarks = [
                    Benchmarks.MEAN_INTER_TIME,
                    Benchmarks.MARK_DISTRIBUTION,
                    Benchmarks.INTERTIME_DISTRIBUTION,
                ]
                benchmark_manager.run(
                    default_benchmarks, all_data_configs, **benchmark_params
                )

            self.print_info(f"Results saved at: {benchmark_manager.base_dir}")

            return True

        except Exception as e:
            self.print_error_with_traceback(f"Error during benchmark: {e}", e)
            if self.debug:
                self.logger.exception("Error details:")
            return False

    def list_available_benchmarks(self) -> List[str]:
        """Return the list of available benchmarks."""
        if Benchmarks is None:
            self.print_error("BenchmarksEnum not available")
            return []

        benchmarks = [benchmark.benchmark_name for benchmark in Benchmarks]

        if self.console:
            from rich.table import Table

            table = Table(title="Available TPP Benchmarks")
            table.add_column("Name", style="cyan")
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
            print("\n=== Available Benchmarks ===")
            for benchmark in benchmarks:
                print(f"- {benchmark}")

        return benchmarks
