"""
Sample Distribution Mark Benchmark

This benchmark creates bins to approximate the distribution of event marks (types) 
from the training dataset, then predicts marks by sampling from this distribution.
"""

import numpy as np
from typing import Dict, Any, Tuple
import torch
import yaml

from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.utils import logger
from .base_bench import BaseBenchmark


class MarkDistributionBenchmark(BaseBenchmark):
    """
    Benchmark that samples event marks from the empirical distribution of training data.
    """
    
    def __init__(self, data_config: DataConfig, experiment_id: str, save_dir: str = None):
        """
        Initialize the mark distribution benchmark.
        
        Args:
            data_config: Data configuration object
            experiment_id: Experiment ID
            save_dir: Directory to save results
        """
        super().__init__(data_config, experiment_id, save_dir)
        
        # Distribution parameters
        self.mark_probabilities = None
        
    @property
    def benchmark_name(self) -> str:
        """Return the name of this benchmark."""
        return "mark_distribution_sampling"
    
    def _prepare_benchmark(self) -> None:
        """
        Build the empirical distribution of event marks from training data.
        """
        train_loader = self.data_module.train_dataloader()
        mark_counts = np.zeros(self.num_event_types, dtype=int)
        total_events = 0
        
        logger.info("Collecting event marks from training data...")
        
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
                    total_events += 1
        
        # Calculate probabilities
        if total_events > 0:
            self.mark_probabilities = mark_counts / total_events
        else:
            # Uniform distribution as fallback
            self.mark_probabilities = np.ones(self.num_event_types) / self.num_event_types
        
        logger.info(f"Collected {total_events} event marks")
        logger.info(f"Event type distribution: {dict(enumerate(self.mark_probabilities))}")
        
    def _sample_from_distribution(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Sample event marks from the empirical distribution.
        
        Args:
            size: Shape of the tensor to generate (batch_size, seq_len)
            
        Returns:
            Tensor of sampled event marks
        """
        total_samples = size[0] * size[1]
        
        # Sample event types according to probabilities
        sampled_marks = np.random.choice(
            self.num_event_types, 
            size=total_samples, 
            p=self.mark_probabilities
        )
        
        # Reshape and convert to tensor
        sampled_tensor = torch.tensor(sampled_marks.reshape(size), dtype=torch.long)
        
        return sampled_tensor
    
    def _create_predictions(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create predictions by sampling from the mark distribution.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (predicted_inter_times, predicted_types)
        """
        # Unpack batch dict
        time_delta_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        
        batch_size, seq_len = type_seqs.shape
        
        # For inter-times, just copy the true values (this benchmark focuses on mark prediction)
        pred_inter_times = time_delta_seqs.clone()
        
        # Sample marks from distribution
        pred_types = self._sample_from_distribution((batch_size, seq_len))
        
        # Move to same device as input
        if type_seqs.device != pred_types.device:
            pred_types = pred_types.to(type_seqs.device)        
        return pred_inter_times, pred_types
    
    def _prepare_benchmark_results(self, aggregated_metrics: Dict[str, float], 
                                  num_batches: int) -> Dict[str, Any]:
        """
        Prepare benchmark-specific results.
        
        Args:
            aggregated_metrics: Aggregated metrics from evaluation
            num_batches: Number of batches evaluated
            
        Returns:
            Dictionary with benchmark-specific results
        """
        results = super()._prepare_results(aggregated_metrics, num_batches)
        
        # Add distribution-specific information
        results['distribution_stats'] = {
            'mark_probabilities': self.mark_probabilities.tolist(),
            'entropy': float(-np.sum(self.mark_probabilities * np.log(self.mark_probabilities + 1e-10)))
        }
        
        return results


def run_mark_distribution_benchmark(config_path: str, experiment_id: str, save_dir: str = None) -> Dict[str, Any]:
    """
    Run the mark distribution sampling benchmark.
    
    Args:
        config_path: Path to configuration file
        experiment_id: Experiment ID in the configuration
        save_dir: Directory to save results
        
    Returns:
        Benchmark results
    """
    # Load DataConfig from YAML
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    data_config = DataConfig.from_dict(config_dict["data_config"])
    
    # Create and run benchmark
    benchmark = MarkDistributionBenchmark(data_config, experiment_id, save_dir)
    results = benchmark.evaluate()
    
    logger.info("Mark Distribution Benchmark completed successfully!")
    logger.info(f"Type Accuracy: {results['metrics'].get('type_accuracy_mean', 'N/A'):.6f}")
    logger.info(f"Macro F1 Score: {results['metrics'].get('macro_f1score_mean', 'N/A'):.6f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Mark Distribution Benchmark")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--experiment_id', type=str, required=True,
                        help='Experiment ID in the configuration')
    parser.add_argument('--save_dir', type=str, default="./benchmark_results",
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    run_mark_distribution_benchmark(args.config_path, args.experiment_id, args.save_dir)