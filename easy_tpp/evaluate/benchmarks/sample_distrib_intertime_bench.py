"""
Sample Distribution Inter-Time Benchmark

This benchmark creates bins to approximate the distribution of inter-times from the 
training dataset, then predicts inter-times by sampling from these bins.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Tuple, List
import torch
import yaml

from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.preprocess.data_loader import TPPDataModule
from easy_tpp.utils import logger
from .base_bench import BaseBenchmark


class InterTimeDistributionBenchmark(BaseBenchmark):
    """
    Benchmark that samples inter-times from the empirical distribution of training data.
    """
    
    def __init__(self, data_config: DataConfig, experiment_id: str, save_dir: str = None, num_bins: int = 50):
        super().__init__(data_config, experiment_id, save_dir)
        self.num_bins = num_bins
        
        # Distribution parameters
        self.bins = None
        self.bin_probabilities = None
        self.bin_centers = None
        
        # Initialize data module
        self.data_module = TPPDataModule(data_config)
        self.data_module.setup('fit')  # Setup train and validation data
        self.data_module.setup('test')  # Setup test data
        
    def _build_intertime_distribution(self) -> None:
        """
        Build the empirical distribution of inter-times from training data.
        """
        train_loader = self.data_module.train_dataloader()
        all_inter_times = []
        
        logger.info("Collecting inter-times from training data...")
        
        for batch in train_loader:
            # Extract inter-times from batch
            # batch structure: dict with keys: 'time_seqs', 'time_delta_seqs', 'type_seqs', 'batch_non_pad_mask', ...
            time_delta_seqs = batch['time_delta_seqs']  # Inter-times
            batch_non_pad_mask = batch.get('batch_non_pad_mask', None)
            
            if batch_non_pad_mask is not None:
                # Only consider non-padded values
                mask = batch_non_pad_mask.bool()
                valid_inter_times = time_delta_seqs[mask]
            else:
                valid_inter_times = time_delta_seqs.flatten()
            
            # Filter out zero or negative inter-times
            valid_inter_times = valid_inter_times[valid_inter_times > 0]
            all_inter_times.extend(valid_inter_times.cpu().numpy().tolist())
        
        all_inter_times = np.array(all_inter_times)
        logger.info(f"Collected {len(all_inter_times)} inter-time samples")
        
        # Create histogram
        counts, bin_edges = np.histogram(all_inter_times, bins=self.num_bins)
        
        # Calculate bin centers
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate probabilities
        self.bin_probabilities = counts / np.sum(counts)
        
        # Store bin edges for reference
        self.bins = bin_edges
        
        logger.info(f"Built distribution with {self.num_bins} bins")
        logger.info(f"Inter-time range: [{np.min(all_inter_times):.6f}, {np.max(all_inter_times):.6f}]")
        
    def _sample_from_distribution(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Sample inter-times from the empirical distribution.
        
        Args:
            size: Shape of the tensor to generate (batch_size, seq_len)
            
        Returns:
            Tensor of sampled inter-times
        """
        total_samples = size[0] * size[1]
        
        # Sample bin indices according to probabilities
        bin_indices = np.random.choice(
            len(self.bin_centers), 
            size=total_samples, 
            p=self.bin_probabilities
        )
        
        # Get values from selected bins (use bin centers)
        sampled_values = self.bin_centers[bin_indices]
        
        # Add some noise within bins for better approximation
        bin_width = self.bins[1] - self.bins[0]  # Assuming uniform bin width
        noise = np.random.uniform(-bin_width/2, bin_width/2, size=total_samples)
        sampled_values += noise
        
        # Ensure positive values
        sampled_values = np.maximum(sampled_values, 1e-6)
        
        # Reshape and convert to tensor
        sampled_tensor = torch.tensor(sampled_values.reshape(size), dtype=torch.float32)
        
        return sampled_tensor
    
    def _create_predictions(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create predictions by sampling from the inter-time distribution.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (predicted_inter_times, predicted_types)
        """
        # Extract batch components
        # Unpack batch dict
        time_delta_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        batch_size, seq_len = time_delta_seqs.shape
        
        # Sample inter-times from distribution
        pred_inter_times = self._sample_from_distribution((batch_size, seq_len))
        
        # Move to same device as input
        if time_delta_seqs.device != pred_inter_times.device:
            pred_inter_times = pred_inter_times.to(time_delta_seqs.device)
        
        # For types, just copy the true types (this benchmark focuses on time prediction)
        pred_types = type_seqs.clone()
        
        return pred_inter_times, pred_types
    
    @property
    def benchmark_name(self) -> str:
        return "intertime_distribution_sampling"

    def _prepare_benchmark(self) -> None:
        self._build_intertime_distribution()

    def _prepare_benchmark_results(self, aggregated_metrics: Dict[str, float], num_batches: int) -> Dict[str, Any]:
        results = super()._prepare_results(aggregated_metrics, num_batches)
        results['distribution_stats'] = {
            'bin_probabilities': self.bin_probabilities.tolist() if self.bin_probabilities is not None else [],
            'bin_centers': self.bin_centers.tolist() if self.bin_centers is not None else [],
        }
        return results
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the inter-time distribution benchmark.
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting Inter-Time Distribution Benchmark evaluation...")
        
        # Build empirical distribution from training data
        self._build_intertime_distribution()
        
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
            'benchmark_name': 'inter_time_distribution_sampling',
            'dataset_name': self.dataset_name,
            'num_bins': self.num_bins,
            'distribution_stats': {
                'bin_centers': self.bin_centers.tolist(),
                'bin_probabilities': self.bin_probabilities.tolist(),
                'bin_edges': self.bins.tolist()
            },
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
        results_file = os.path.join(dataset_dir, "sample_distrib_intertime_bench_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")


def run_intertime_distribution_benchmark(config_path: str, experiment_id: str, save_dir: str = None, num_bins: int = 50) -> Dict[str, Any]:
    """
    Run the inter-time distribution sampling benchmark.
    
    Args:
        config_path: Path to configuration file
        experiment_id: Experiment ID in the configuration
        save_dir: Directory to save results
        num_bins: Number of bins for histogram approximation
        
    Returns:
        Benchmark results
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    data_config = DataConfig.from_dict(config_dict["data_config"])
    benchmark = InterTimeDistributionBenchmark(data_config, experiment_id, save_dir, num_bins)
    results = benchmark.evaluate()
    
    logger.info("Inter-Time Distribution Benchmark completed successfully!")
    logger.info(f"Time RMSE: {results['metrics'].get('time_rmse_mean', 'N/A'):.6f}")
    logger.info(f"Time MAE: {results['metrics'].get('time_mae_mean', 'N/A'):.6f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Inter-Time Distribution Benchmark")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--experiment_id', type=str, required=True,
                        help='Experiment ID in the configuration')
    parser.add_argument('--save_dir', type=str, default="./benchmark_results",
                        help='Directory to save results')
    parser.add_argument('--num_bins', type=int, default=50,
                        help='Number of bins for histogram approximation')
    
    args = parser.parse_args()
    
    run_intertime_distribution_benchmark(
        args.config_path, args.experiment_id, args.save_dir, args.num_bins
    )