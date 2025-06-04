"""
Mean Benchmark for Inter-Time Prediction

This benchmark always predicts the mean inter-time from the training dataset.
It computes RMSE and other time-based metrics using the metrics helper.
"""

import numpy as np
from typing import Dict, Any, Tuple
import torch
import yaml

from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.utils import logger
from .base_bench import BaseBenchmark, run_benchmark


class MeanInterTimeBenchmark(BaseBenchmark):
    """
    Benchmark that predicts the mean inter-time for all events.
    """
    
    def __init__(self, data_config: DataConfig, experiment_id: str, save_dir: str = None):
        """
        Initialize the mean inter-time benchmark.
        
        Args:
            data_config: Data configuration object
            experiment_id: Experiment ID
            save_dir: Directory to save results
        """
        super().__init__(data_config, experiment_id, save_dir)
        self.mean_inter_time = None
    
    @property
    def benchmark_name(self) -> str:
        """Return the name of this benchmark."""
        return "mean_inter_time"
        
    def _prepare_benchmark(self) -> None:
        """
        Compute the mean inter-time from the training dataset.
        """
        train_loader = self.data_module.train_dataloader()
        all_inter_times = []
        
        logger.info("Computing mean inter-time from training data...")
        
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
            
            all_inter_times.extend(valid_inter_times.cpu().numpy().tolist())
        
        self.mean_inter_time = np.mean(all_inter_times)
        logger.info(f"Computed mean inter-time: {self.mean_inter_time:.6f}")
    
    def _create_predictions(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create predictions using the mean inter-time.
        
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
        
        # Create predictions with mean inter-time
        pred_inter_times = torch.full_like(time_delta_seqs, self.mean_inter_time)
        
        # For types, just copy the true types (this benchmark focuses on time prediction)
        pred_types = type_seqs.clone()
        
        return pred_inter_times, pred_types
    
    def _get_custom_results_info(self) -> Dict[str, Any]:
        """Add custom information specific to this benchmark."""
        return {
            'mean_inter_time_used': float(self.mean_inter_time) if self.mean_inter_time is not None else None
        }


def run_mean_benchmark(config_path: str, experiment_id: str, save_dir: str = None) -> Dict[str, Any]:
    """
    Run the mean inter-time benchmark.
    
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
    return run_benchmark(MeanInterTimeBenchmark, data_config, experiment_id, save_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Mean Inter-Time Benchmark")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--experiment_id', type=str, required=True,
                        help='Experiment ID in the configuration')
    parser.add_argument('--save_dir', type=str, default="./benchmark_results",
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    run_mean_benchmark(args.config_path, args.experiment_id, args.save_dir)