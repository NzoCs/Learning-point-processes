"""
Base Benchmark Class

This module provides a base class for implementing benchmarks for TPP models.
It defines the common interface and shared functionality that all benchmarks should implement.
"""

import os
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import torch
import yaml

from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.preprocess.data_loader import TPPDataModule
from easy_tpp.evaluate.metrics_helper import MetricsHelper, EvaluationMode
from easy_tpp.utils import logger


class BaseBenchmark(ABC):
    """
    Abstract base class for TPP benchmarks.
    
    This class provides the common structure and functionality that all benchmarks
    should inherit from. It handles data loading, metrics computation, result
    aggregation, and file saving.
    """
    def __init__(self, data_config: DataConfig, experiment_id: str, save_dir: str = None):
        """
        Initialize the base benchmark.
        
        Args:
            data_config: DataConfig object
            experiment_id: Name/ID of the experiment or dataset
            save_dir: Directory to save results
        """
        self.data_config = data_config
        self.save_dir = save_dir or "./benchmark_results"
        self.dataset_name = experiment_id
        
        # Initialize data module with data_config
        self.data_module = TPPDataModule(data_config)
        self.data_module.setup('fit')  # Setup train and validation data
        self.data_module.setup('test')  # Setup test data
        
        # Initialize metrics helper
        self.metrics_helper = MetricsHelper(
            num_event_types=self.data_module.num_event_types,
            mode=EvaluationMode.PREDICTION
        )
        
        self.num_event_types = self.data_module.num_event_types
        
    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """
        Return the name of this benchmark.
        
        Returns:
            String identifier for this benchmark
        """
        pass
    
    @abstractmethod
    def _prepare_benchmark(self) -> None:
        """
        Prepare the benchmark by computing any necessary statistics or parameters
        from the training data. This method should be implemented by subclasses
        to perform benchmark-specific preparation.
        """
        pass
    
    @abstractmethod
    def _create_predictions(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create predictions for a given batch using the benchmark strategy.
        
        Args:
            batch: Input batch from the data loader
            
        Returns:
            Tuple of (predicted_inter_times, predicted_types)
        """
        pass
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run the benchmark evaluation.
        
        This method orchestrates the entire benchmark process:
        1. Prepare the benchmark (compute statistics, etc.)
        2. Evaluate on test data
        3. Aggregate metrics
        4. Save results
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting {self.benchmark_name} benchmark evaluation...")
        
        # Prepare benchmark-specific parameters
        self._prepare_benchmark()
        
        # Evaluate on test data
        test_loader = self.data_module.test_dataloader()
        all_metrics = []
        
        for batch_idx, batch in enumerate(test_loader):
            # Create predictions using benchmark strategy
            pred_inter_times, pred_types = self._create_predictions(batch)
            predictions = (pred_inter_times, pred_types)
            
            # Compute metrics
            metrics = self.metrics_helper.compute_all_metrics(batch, predictions)
            all_metrics.append(metrics)
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")
        
        # Aggregate metrics across all batches
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        # Prepare results
        results = self._prepare_results(aggregated_metrics, len(all_metrics))
        
        # Save results
        self._save_results(results)
        
        # Log summary
        self._log_summary(results)
        
        return results
    
    def _aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across all batches.
        
        Args:
            all_metrics: List of metric dictionaries from each batch
            
        Returns:
            Aggregated metrics with mean, std, min, max for each metric
        """
        if not all_metrics:
            return {}
        
        # Get all metric names
        metric_names = all_metrics[0].keys()
        aggregated = {}
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if not np.isnan(m[metric_name])]
            if values:
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[f"{metric_name}_std"] = float(np.std(values))
                aggregated[f"{metric_name}_min"] = float(np.min(values))
                aggregated[f"{metric_name}_max"] = float(np.max(values))
            else:
                aggregated[f"{metric_name}_mean"] = float('nan')
                aggregated[f"{metric_name}_std"] = float('nan')
                aggregated[f"{metric_name}_min"] = float('nan')
                aggregated[f"{metric_name}_max"] = float('nan')
        
        return aggregated
    
    def _prepare_results(self, aggregated_metrics: Dict[str, float], 
                        num_batches: int) -> Dict[str, Any]:
        """
        Prepare the results dictionary with benchmark information.
        
        Args:
            aggregated_metrics: Aggregated metrics
            num_batches: Number of batches processed
            
        Returns:
            Results dictionary
        """
        results = {
            'benchmark_name': self.benchmark_name,
            'dataset_name': self.dataset_name,
            'num_event_types': self.num_event_types,
            'metrics': aggregated_metrics,
            'num_batches_evaluated': num_batches
        }
        
        # Allow subclasses to add custom information
        custom_info = self._get_custom_results_info()
        if custom_info:
            results.update(custom_info)
        
        return results
    
    def _get_custom_results_info(self) -> Dict[str, Any]:
        """
        Get custom information to add to results.
        Subclasses can override this to add benchmark-specific information.
        
        Returns:
            Dictionary with custom information
        """
        return {}
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Results dictionary to save
        """
        # Create save directory
        dataset_dir = os.path.join(self.save_dir, self.dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(dataset_dir, f"{self.benchmark_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    def _log_summary(self, results: Dict[str, Any]) -> None:
        """
        Log a summary of the benchmark results.
        
        Args:
            results: Results dictionary
        """
        logger.info(f"{self.benchmark_name} benchmark completed successfully!")
        
        metrics = results.get('metrics', {})
        
        # Log time-based metrics if available
        if 'time_rmse_mean' in metrics:
            logger.info(f"Time RMSE: {metrics['time_rmse_mean']:.6f}")
        if 'time_mae_mean' in metrics:
            logger.info(f"Time MAE: {metrics['time_mae_mean']:.6f}")
            
        # Log type-based metrics if available
        if 'type_accuracy_mean' in metrics:
            logger.info(f"Type Accuracy: {metrics['type_accuracy_mean']:.6f}")
        if 'macro_f1score_mean' in metrics:
            logger.info(f"Macro F1 Score: {metrics['macro_f1score_mean']:.6f}")
    
    def get_available_metrics(self) -> List[str]:
        """
        Get the list of available metrics for this benchmark.
        
        Returns:
            List of metric names
        """
        return self.metrics_helper.get_available_metrics()


def run_benchmark(benchmark_class, config_path: str, experiment_id: str, 
                 save_dir: str = None, **kwargs) -> Dict[str, Any]:
    """
    Generic function to run any benchmark.
    
    Args:
        benchmark_class: The benchmark class to instantiate
        config_path: Path to configuration file
        experiment_id: Experiment ID in the configuration
        save_dir: Directory to save results
        **kwargs: Additional arguments to pass to benchmark constructor
        
    Returns:
        Benchmark results
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    from easy_tpp.config_factory.data_config import DataConfig
    data_config = DataConfig.from_dict(config_dict["data_config"])
    benchmark = benchmark_class(data_config, experiment_id, save_dir, **kwargs)
    results = benchmark.evaluate()
    
    return results
