"""
Last Mark Benchmark

This benchmark always predicts the previous event mark (type) as the next mark.
For the first event in a sequence, it uses the most common mark from training data.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Tuple, List
import torch
import yaml

from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.utils import logger
from .base_bench import BaseBenchmark


class LastMarkBenchmark(BaseBenchmark):
    """
    Benchmark that predicts the previous event mark as the next mark.
    """
    
    def __init__(self, data_config: DataConfig, experiment_id: str, save_dir: str = None):
        super().__init__(data_config, experiment_id, save_dir)
        self.most_common_mark = None
        
    def _find_most_common_mark(self) -> int:
        """
        Find the most common event mark in the training data.
        This will be used as a fallback for the first event in sequences.
        
        Returns:
            Most common event mark
        """
        train_loader = self.data_module.train_dataloader()
        mark_counts = np.zeros(self.num_event_types, dtype=int)
        
        logger.info("Finding most common mark from training data...")
        
        for batch in train_loader:
            # Extract event types from batch
            # batch structure: dict with keys: 'time_seqs', 'time_delta_seqs', 'type_seqs', 'batch_non_pad_mask', ...
            type_seqs = batch['type_seqs']  # Event types/marks
            batch_non_pad_mask = batch.get('batch_non_pad_mask', None)
            
            if batch_non_pad_mask is not None:
                # Only consider non-padded values
                mask = batch_non_pad_mask.bool()
                valid_types = type_seqs[mask]
            else:
                valid_types = type_seqs.flatten()
            
            # Convert to numpy and count occurrences
            valid_types_np = valid_types.cpu().numpy().astype(int)
            
            # Count each event type
            for event_type in valid_types_np:
                if 0 <= event_type < self.num_event_types:
                    mark_counts[event_type] += 1
        
        # Find most common mark
        most_common_mark = np.argmax(mark_counts)
        logger.info(f"Most common mark: {most_common_mark} (count: {mark_counts[most_common_mark]})")
        
        return int(most_common_mark)
    
    def _create_predictions(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create predictions using the last mark strategy.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (predicted_inter_times, predicted_types)
        """
        # Extract batch components
        # Unpack batch dict
        time_delta_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        batch_size, seq_len = type_seqs.shape
        
        # For inter-times, just copy the true values (this benchmark focuses on mark prediction)
        pred_inter_times = time_delta_seqs.clone()
        
        # Create predictions for marks using "last mark" strategy
        pred_types = torch.zeros_like(type_seqs)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if j == 0:
                    # For first event, use most common mark
                    pred_types[i, j] = self.most_common_mark
                else:
                    # For subsequent events, use previous mark
                    pred_types[i, j] = type_seqs[i, j-1]
        
        return pred_inter_times, pred_types
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the last mark benchmark.
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting Last Mark Benchmark evaluation...")
        
        # Find most common mark from training data
        self.most_common_mark = self._find_most_common_mark()
        
        # Evaluate on test data
        test_loader = self.data_module.test_dataloader()
        all_metrics = []
        
        for batch_idx, batch in enumerate(test_loader):
            # Create predictions
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
        results = {
            'benchmark_name': 'last_mark',
            'dataset_name': self.dataset_name,
            'num_event_types': self.num_event_types,
            'most_common_mark_used': int(self.most_common_mark),
            'strategy': 'predict_previous_mark',
            'metrics': aggregated_metrics,
            'num_batches_evaluated': len(all_metrics)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across all batches.
        
        Args:
            all_metrics: List of metric dictionaries
            
        Returns:
            Aggregated metrics
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
        results_file = os.path.join(dataset_dir, "last_mark_bench_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    @property
    def benchmark_name(self) -> str:
        return "last_mark_benchmark"

    def _prepare_benchmark(self) -> None:
        self.most_common_mark = 0  # Pour le test, on évite l'accès disque

    def _prepare_benchmark_results(self, aggregated_metrics: Dict[str, float], num_batches: int) -> Dict[str, Any]:
        results = super()._prepare_results(aggregated_metrics, num_batches)
        results['most_common_mark'] = self.most_common_mark
        return results


def run_last_mark_benchmark(config_path: str, experiment_id: str, save_dir: str = None) -> Dict[str, Any]:
    """
    Run the last mark benchmark.
    
    Args:
        config_path: Path to configuration file
        experiment_id: Experiment ID in the configuration
        save_dir: Directory to save results
        
    Returns:
        Benchmark results
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    data_config = DataConfig.from_dict(config_dict["data_config"])
    benchmark = LastMarkBenchmark(data_config, experiment_id, save_dir)
    results = benchmark.evaluate()
    
    logger.info("Last Mark Benchmark completed successfully!")
    logger.info(f"Type Accuracy: {results['metrics'].get('type_accuracy_mean', 'N/A'):.6f}")
    logger.info(f"Macro F1 Score: {results['metrics'].get('macro_f1score_mean', 'N/A'):.6f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Last Mark Benchmark")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--experiment_id', type=str, required=True,
                        help='Experiment ID in the configuration')
    parser.add_argument('--save_dir', type=str, default="./benchmark_results",
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    run_last_mark_benchmark(args.config_path, args.experiment_id, args.save_dir)