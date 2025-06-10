"""
Last Mark Benchmark

This benchmark predicts the next event mark (type) using the previous event mark (lag-1).
"""

from typing import Dict, Any, Tuple
import torch
import yaml

from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.utils import logger
from .base_bench import BaseBenchmark, BenchmarkMode


class LastMarkBenchmark(BaseBenchmark):
    """
    Benchmark that predicts the previous event mark as the next mark (lag-1).
    """
    
    def __init__(self, data_config: DataConfig, experiment_id: str, save_dir: str = None):
        """
        Initialize the last mark benchmark.
        Args:
            data_config: Data configuration object
            experiment_id: Experiment ID
            save_dir: Directory to save results
        """
        super().__init__(data_config, experiment_id, save_dir, benchmark_mode=BenchmarkMode.TYPE_ONLY)

    def _create_type_predictions(self, batch: Tuple) -> torch.Tensor:
        """
        Create type predictions using the lag-1 mark strategy.
        
        Args:
            batch: Input batch
            
        Returns:
            Tensor of predicted types
        """
        type_seqs = batch['type_seqs']
        batch_size, seq_len = type_seqs.shape
        
        # Create predictions for marks using lag-1 strategy
        pred_types = torch.zeros_like(type_seqs)
        safe_type_seqs = type_seqs.masked_fill(type_seqs == self.pad_token, 0)  # Avoid pad token issues

        # For positions 1 to seq_len-1, use the previous mark (lag-1)
        if seq_len > 1:
            pred_types[:, 1:] = safe_type_seqs[:, :-1]

        # For the first position (index 0), we cannot predict, so it will remain 0
        
        return pred_types
    
    @property
    def benchmark_name(self) -> str:
        return "lag1_mark_benchmark"

    def _prepare_benchmark(self) -> None:
        pass  # No special preparation needed for lag-1 strategy

    def _get_custom_results_info(self) -> Dict[str, Any]:
        """Add custom information specific to this benchmark."""
        return {
            'strategy': 'lag1_mark_prediction'
        }


def run_last_mark_benchmark(config_path: str, experiment_id: str, save_dir: str = None) -> Dict[str, Any]:
    """
    Run the lag-1 mark benchmark.
    
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
    
    logger.info("Lag-1 Mark Benchmark completed successfully!")
    logger.info(f"Type Accuracy: {results['metrics'].get('type_accuracy_mean', 'N/A'):.6f}")
    logger.info(f"Macro F1 Score: {results['metrics'].get('macro_f1score_mean', 'N/A'):.6f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Lag-1 Mark Benchmark")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--experiment_id', type=str, required=True,
                        help='Experiment ID in the configuration')
    parser.add_argument('--save_dir', type=str, default="./benchmark_results",
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    run_last_mark_benchmark(args.config_path, args.experiment_id, args.save_dir)