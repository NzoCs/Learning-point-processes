"""
Sample Distribution Mark Benchmark

This benchmark creates bins to approximate the distribution of event marks (types)
from the training dataset, then predicts marks by sampling from this distribution.
"""

from typing import Dict, Any, Tuple
import torch
import yaml

from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.utils import logger
from .base_bench import BaseBenchmark, BenchmarkMode


class MarkDistributionBenchmark(BaseBenchmark):
    """
    Benchmark that samples event marks from the empirical distribution of training data.
    """

    def __init__(
        self, data_config: DataConfig, experiment_id: str, save_dir: str = None
    ):
        """
        Initialize the mark distribution benchmark.

        Args:
            data_config: Data configuration object
            experiment_id: Experiment ID
            save_dir: Directory to save results
        """
        # This benchmark focuses on type prediction, so default to TYPE_ONLY
        super().__init__(
            data_config, experiment_id, save_dir, benchmark_mode=BenchmarkMode.TYPE_ONLY
        )

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

        test_loader = self.data_module.test_dataloader()
        mark_counts = torch.zeros(self.num_event_types, dtype=torch.int64)
        total_events = 0

        logger.info("Collecting event marks from test data...")

        for batch in test_loader:
            # Extract event types from batch
            # batch structure: dict with keys: 'time_seqs', 'time_delta_seqs', 'type_seqs', 'batch_non_pad_mask', ...
            type_seqs = batch["type_seqs"]  # Event types/marks
            batch_non_pad_mask = batch.get("batch_non_pad_mask", None)

            if batch_non_pad_mask is not None:
                # Only consider non-padded values
                mask = batch_non_pad_mask.bool()
                valid_types = type_seqs[mask]
            else:
                valid_types = type_seqs.flatten()
            # Count each event type using PyTorch
            valid_types = valid_types.long()  # Ensure integer type

            # Filter valid types
            valid_mask = (valid_types >= 0) & (valid_types < self.num_event_types)
            valid_types_filtered = valid_types[valid_mask]

            # Count occurrences using bincount
            if len(valid_types_filtered) > 0:
                counts = torch.bincount(
                    valid_types_filtered, minlength=self.num_event_types
                )
                mark_counts += counts
                total_events += len(valid_types_filtered)
        # Calculate probabilities
        if total_events > 0:
            self.mark_probabilities = mark_counts.float() / total_events
        else:
            # Uniform distribution as fallback
            self.mark_probabilities = (
                torch.ones(self.num_event_types) / self.num_event_types
            )

        logger.info(f"Collected {total_events} event marks")
        logger.info(
            f"Event type distribution: {dict(enumerate(self.mark_probabilities))}"
        )

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
        sampled_marks = torch.multinomial(
            self.mark_probabilities, num_samples=total_samples, replacement=True
        )

        # Reshape and convert to tensor
        sampled_tensor = sampled_marks.reshape(size)

        return sampled_tensor

    def _create_type_predictions(self, batch: Tuple) -> torch.Tensor:
        """
        Create type predictions by sampling from the mark distribution.

        Args:
            batch: Input batch

        Returns:
            Tensor of predicted types
        """
        type_seqs = batch["type_seqs"]
        batch_size, seq_len = type_seqs.shape

        # Sample marks from distribution
        pred_types = self._sample_from_distribution((batch_size, seq_len))

        # Move to same device as input
        if type_seqs.device != pred_types.device:
            pred_types = pred_types.to(type_seqs.device)

        return pred_types

    def _get_custom_results_info(self) -> Dict[str, Any]:
        """Get custom information to add to results."""
        return {
            "distribution_stats": {
                "entropy": (
                    float(
                        -torch.sum(
                            self.mark_probabilities
                            * torch.log(self.mark_probabilities + 1e-10)
                        ).item()
                    )
                    if self.mark_probabilities is not None
                    else 0.0
                )
            }
        }


def run_mark_distribution_benchmark(
    config_path: str, experiment_id: str, save_dir: str = None
) -> Dict[str, Any]:
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
    logger.info(
        f"Type Accuracy: {results['metrics'].get('type_accuracy_mean', 'N/A'):.6f}"
    )
    logger.info(
        f"Macro F1 Score: {results['metrics'].get('macro_f1score_mean', 'N/A'):.6f}"
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Mark Distribution Benchmark")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="Experiment ID in the configuration",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    run_mark_distribution_benchmark(args.config_path, args.experiment_id, args.save_dir)
