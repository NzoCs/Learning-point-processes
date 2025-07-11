#!/usr/bin/env python3
"""
Outil en ligne de commande pour exécuter les benchmarks EasyTPP

Ce script permet d'exécuter différents benchmarks sur les datasets configurés
dans bench_config.yaml. Il supporte l'exécution de tous les benchmarks ou
de benchmarks spécifiques sur des datasets spécifiques.

Usage:
    python run_bench.py --help
    python run_bench.py --list-datasets
    python run_bench.py --list-benchmarks
    python run_bench.py --dataset taxi --benchmark mean
    python run_bench.py --dataset taxi --all-benchmarks
    python run_bench.py --all-datasets --benchmark mean
    python run_bench.py --all-datasets --all-benchmarks
"""

import argparse
import os
import sys
import yaml
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.evaluation.benchmarks.mean_bench import MeanInterTimeBenchmark
from easy_tpp.evaluation.benchmarks.sample_distrib_mark_bench import MarkDistributionBenchmark
from easy_tpp.evaluation.benchmarks.sample_distrib_intertime_bench import InterTimeDistributionBenchmark
from easy_tpp.evaluation.benchmarks.last_mark_bench import LastMarkBenchmark
from easy_tpp.utils import logger


@dataclass
class BenchmarkInfo:
    """Information about a benchmark."""
    name: str
    class_ref: Any
    description: str


class BenchmarkRunner:
    """Runner for executing benchmarks on configured datasets."""
    
    # Available benchmarks
    AVAILABLE_BENCHMARKS = {
        'mean': BenchmarkInfo(
            name='mean',
            class_ref=MeanInterTimeBenchmark,
            description='Mean Inter-Time Benchmark - predicts mean inter-arrival time'
        ),
        'mark_distribution': BenchmarkInfo(
            name='mark_distribution', 
            class_ref=MarkDistributionBenchmark,
            description='Mark Distribution Benchmark - samples marks from training distribution'
        ),
        'intertime_distribution': BenchmarkInfo(
            name='intertime_distribution',
            class_ref=InterTimeDistributionBenchmark,
            description='Inter-Time Distribution Benchmark - samples inter-times from training distribution'
        ),
        'last_mark': BenchmarkInfo(
            name='last_mark',
            class_ref=LastMarkBenchmark,
            description='Last Mark Benchmark - predicts the last observed mark'
        )
    }
    
    def __init__(self, config_path: str = None, output_dir: str = None, debug: bool = False):
        """
        Initialize the benchmark runner.
        
        Args:
            config_path: Path to bench_config.yaml file
            output_dir: Output directory for results
            debug: Enable debug mode to capture more detailed error information
        """
        if config_path is None:
            config_path = Path(__file__).parent / "bench_config.yaml"
        
        self.config_path = Path(config_path)
        self.output_dir = output_dir or "./benchmark_results"
        self.debug = debug
        
        # Load configuration
        self.config = self._load_config()
        self.datasets = self.config.get('data', {})
        
        logger.info(f"Configuration loaded from: {self.config_path}")
        logger.info(f"Found {len(self.datasets)} datasets: {list(self.datasets.keys())}")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load benchmark configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _create_data_config(self, dataset_name: str) -> DataConfig:
        """
        Create DataConfig from dataset configuration.
        
        Args:
            dataset_name: Name of the dataset in config
            
        Returns:
            DataConfig object
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        dataset_config = self.datasets[dataset_name].copy()
        
        # Add required fields for DataConfig
        dataset_config['dataset_id'] = dataset_name
        dataset_config['data_loading_specs'] = dataset_config.get('data_loading_specs', {
            'batch_size': 32,
            'num_workers': 1,
            'tensor_type': 'pt'
        })
        
        return DataConfig.from_dict(dataset_config)
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.datasets.keys())
    
    def list_benchmarks(self) -> List[str]:
        """List all available benchmarks."""
        return list(self.AVAILABLE_BENCHMARKS.keys())
    
    def get_benchmark_info(self, benchmark_name: str) -> BenchmarkInfo:
        """Get information about a specific benchmark."""
        if benchmark_name not in self.AVAILABLE_BENCHMARKS:
            raise ValueError(f"Benchmark '{benchmark_name}' not found")
        return self.AVAILABLE_BENCHMARKS[benchmark_name]
    
    def run_benchmark(self, dataset_name: str, benchmark_name: str) -> Dict[str, Any]:
        """
        Run a specific benchmark on a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            benchmark_name: Name of the benchmark
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running {benchmark_name} benchmark on {dataset_name} dataset...")
        
        # Validate inputs
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        if benchmark_name not in self.AVAILABLE_BENCHMARKS:
            raise ValueError(f"Benchmark '{benchmark_name}' not found")
        
        # Create data config
        data_config = self._create_data_config(dataset_name)
        
        # Get benchmark class
        benchmark_info = self.AVAILABLE_BENCHMARKS[benchmark_name]
        benchmark_class = benchmark_info.class_ref
        
        # Create and run benchmark
        start_time = time.time()
        try:
            benchmark = benchmark_class(
                data_config=data_config,
                experiment_id=dataset_name,
                save_dir=self.output_dir
            )
            
            if self.debug:
                logger.info(f"Debug mode enabled: Detailed error information will be shown")
            
            results = benchmark.evaluate()
            
            execution_time = time.time() - start_time
            results['execution_time_seconds'] = execution_time
            
            logger.info(f"✅ {benchmark_name} benchmark completed successfully on {dataset_name}")
            logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {benchmark_name} benchmark failed on {dataset_name}: {str(e)}")
            logger.error(f"⏱️  Failed after: {execution_time:.2f} seconds")
            
            if self.debug:
                logger.error("Detailed error information:")
                import traceback
                traceback.print_exc()
            
            raise
    
    def run_all_benchmarks_on_dataset(self, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Run all benchmarks on a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with results for each benchmark
        """
        logger.info(f"Running ALL benchmarks on {dataset_name} dataset...")
        
        all_results = {}
        failed_benchmarks = []
        
        for benchmark_name in self.AVAILABLE_BENCHMARKS.keys():
            try:
                results = self.run_benchmark(dataset_name, benchmark_name)
                all_results[benchmark_name] = results
            except Exception as e:
                logger.error(f"Failed to run {benchmark_name} on {dataset_name}: {str(e)}")
                failed_benchmarks.append(benchmark_name)
                continue
        
        logger.info(f"✅ Completed {len(all_results)} benchmarks on {dataset_name}")
        if failed_benchmarks:
            logger.warning(f"⚠️  Failed benchmarks: {failed_benchmarks}")
        
        return all_results
    
    def run_benchmark_on_all_datasets(self, benchmark_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Run a specific benchmark on all datasets.
        
        Args:
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with results for each dataset
        """
        logger.info(f"Running {benchmark_name} benchmark on ALL datasets...")
        
        all_results = {}
        failed_datasets = []
        
        for dataset_name in self.datasets.keys():
            try:
                results = self.run_benchmark(dataset_name, benchmark_name)
                all_results[dataset_name] = results
            except Exception as e:
                logger.error(f"Failed to run {benchmark_name} on {dataset_name}: {str(e)}")
                failed_datasets.append(dataset_name)
                continue
        
        logger.info(f"✅ Completed {benchmark_name} benchmark on {len(all_results)} datasets")
        if failed_datasets:
            logger.warning(f"⚠️  Failed datasets: {failed_datasets}")
        
        return all_results
    
    def run_all_benchmarks_on_all_datasets(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run all benchmarks on all datasets.
        
        Returns:
            Nested dictionary with results: {dataset_name: {benchmark_name: results}}
        """
        logger.info("Running ALL benchmarks on ALL datasets...")
        
        all_results = {}
        total_runs = len(self.datasets) * len(self.AVAILABLE_BENCHMARKS)
        completed_runs = 0
        failed_runs = 0
        
        for dataset_name in self.datasets.keys():
            dataset_results = {}
            
            for benchmark_name in self.AVAILABLE_BENCHMARKS.keys():
                try:
                    results = self.run_benchmark(dataset_name, benchmark_name)
                    dataset_results[benchmark_name] = results
                    completed_runs += 1
                except Exception as e:
                    logger.error(f"Failed {benchmark_name} on {dataset_name}: {str(e)}")
                    failed_runs += 1
                    continue
                
                logger.info(f"Progress: {completed_runs + failed_runs}/{total_runs} runs completed")
            
            if dataset_results:  # Only add if at least one benchmark succeeded
                all_results[dataset_name] = dataset_results
        
        logger.info(f"✅ Completed {completed_runs}/{total_runs} benchmark runs")
        logger.info(f"❌ Failed {failed_runs}/{total_runs} benchmark runs")
        
        return all_results


def print_datasets(runner: BenchmarkRunner):
    """Print available datasets."""
    datasets = runner.list_datasets()
    print("\n📊 Available Datasets:")
    print("=" * 50)
    
    for dataset_name in datasets:
        dataset_config = runner.datasets[dataset_name]
        num_event_types = dataset_config.get('data_specs', {}).get('num_event_types', 'N/A')
        data_format = dataset_config.get('data_format', 'N/A')
        print(f"  • {dataset_name:<15} | Format: {data_format:<6} | Event Types: {num_event_types}")
    print()


def print_benchmarks(runner: BenchmarkRunner):
    """Print available benchmarks."""
    benchmarks = runner.list_benchmarks()
    print("\n🎯 Available Benchmarks:")
    print("=" * 80)
    
    for benchmark_name in benchmarks:
        info = runner.get_benchmark_info(benchmark_name)
        print(f"  • {benchmark_name:<20} | {info.description}")
    print()


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Outil pour exécuter les benchmarks EasyTPP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --list-datasets                           # Afficher les datasets disponibles
  %(prog)s --list-benchmarks                         # Afficher les benchmarks disponibles
  %(prog)s --dataset taxi --benchmark mean           # Exécuter un benchmark spécifique
  %(prog)s --dataset taxi --all-benchmarks           # Tous les benchmarks sur un dataset
  %(prog)s --all-datasets --benchmark mean           # Un benchmark sur tous les datasets
  %(prog)s --all-datasets --all-benchmarks           # Tous les benchmarks sur tous les datasets
  %(prog)s --config custom_config.yaml --output ./results  # Utiliser une config personnalisée
        """
    )
    
    # Configuration files
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to bench_config.yaml file (default: bench_config.yaml in script directory)')
    parser.add_argument('--output', '-o', type=str, default='./benchmark_results',
                        help='Output directory for results (default: ./benchmark_results)')
    
    # Information commands
    parser.add_argument('--list-datasets', action='store_true',
                        help='List all available datasets')
    parser.add_argument('--list-benchmarks', action='store_true',
                        help='List all available benchmarks')      # Execution options
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Dataset name to run benchmark on (default: test if not using --all-datasets)')
    parser.add_argument('--benchmark', '-b', type=str,
                        help='Benchmark name to run')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run on all available datasets')
    parser.add_argument('--all-benchmarks', action='store_true',
                        help='Run all available benchmarks')
    
    # Verbose output
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode for detailed error information')
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = BenchmarkRunner(config_path=args.config, output_dir=args.output, debug=args.debug)
        
        # Handle information commands
        if args.list_datasets:
            print_datasets(runner)
            return
        
        if args.list_benchmarks:
            print_benchmarks(runner)
            return        # Validate execution arguments
        # Check if user explicitly specified both --dataset and --all-datasets
        dataset_specified = '--dataset' in sys.argv or '-d' in sys.argv
        if args.all_datasets and dataset_specified:
            print("❌ Erreur: Vous ne pouvez pas utiliser --dataset et --all-datasets en même temps")
            return 1
        
        # Set default dataset if neither --dataset nor --all-datasets specified
        if not args.all_datasets and args.dataset is None:
            args.dataset = 'test'
            logger.info("No dataset specified, defaulting to 'test' dataset")
        
        if not (args.benchmark or args.all_benchmarks):
            print("❌ Erreur: Vous devez spécifier --benchmark ou --all-benchmarks")
            print("Utilisez --help pour voir l'aide complète")
            return 1
        
        if args.benchmark and args.all_benchmarks:
            print("❌ Erreur: Vous ne pouvez pas utiliser --benchmark et --all-benchmarks en même temps")
            return 1
        
        # Execute benchmarks
        start_time = time.time()
        
        if args.all_datasets and args.benchmark:
            # Single benchmark on all datasets
            results = runner.run_benchmark_on_all_datasets(args.benchmark)
            
        elif args.all_datasets and args.all_benchmarks:
            # All benchmarks on all datasets
            results = runner.run_all_benchmarks_on_all_datasets()
            
        elif args.dataset and args.benchmark:
            # Single benchmark on single dataset
            results = runner.run_benchmark(args.dataset, args.benchmark)
            
        elif args.dataset and args.all_benchmarks:
            # All benchmarks on single dataset
            results = runner.run_all_benchmarks_on_dataset(args.dataset)
        
        else:
            print("❌ Erreur: Combinaison d'arguments non valide")
            print("Utilisez --help pour voir l'aide complète")
            return 1
        
        total_time = time.time() - start_time
        
        logger.info(f"🎉 Benchmark execution completed!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
        logger.info(f"📁 Results saved to: {runner.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏸️  Benchmark execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())