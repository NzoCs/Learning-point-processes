"""
Sample Distribution Inter-Time Benchmark

This benchmark creates bins to approximate the distribution of inter-times from the
training dataset, then predicts inter-times by sampling from these bins.
"""

from typing import Any, Dict, Tuple

import torch
import yaml

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.utils import logger

from .base_bench import Benchmark, BenchmarkMode


class InterTimeDistributionBenchmark(Benchmark):
    """
    Benchmark that samples inter-times from the empirical distribution of training data.
    """

    def __init__(
        self,
        data_config: DataConfig,
        save_dir: str = None,
        num_bins: int = 50,
    ):
        """
        Initialize the inter-time distribution benchmark.

        Args:
            data_config: Data configuration object
*            save_dir: Directory to save results
            num_bins: Number of bins for histogram approximation
        """
        # This benchmark focuses on time prediction, so default to TIME_ONLY
        super().__init__(
            data_config, save_dir, benchmark_mode=BenchmarkMode.TIME_ONLY
        )
        self.num_bins = num_bins

        # Distribution parameters
        self.bins = None
        self.bin_probabilities = None
        self.bin_centers = None

    def _build_intertime_distribution(self) -> None:
        """
        Build the empirical distribution of inter-times from training data.
        """
        train_loader = self.data_module.test_dataloader()
        all_inter_times = []

        logger.info("Collecting inter-times from training data...")

        for batch in train_loader:
            # Extract inter-times from batch
            # batch structure: dict with keys: 'time_seqs', 'time_delta_seqs', 'type_seqs', 'batch_non_pad_mask', ...
            time_delta_seqs = batch["time_delta_seqs"]  # Inter-times
            batch_non_pad_mask = batch.get("batch_non_pad_mask", None)

            if batch_non_pad_mask is not None:
                # Only consider non-padded values
                mask = batch_non_pad_mask.bool()
                valid_inter_times = time_delta_seqs[mask]
            else:
                valid_inter_times = time_delta_seqs.flatten()

            # Filter out zero or negative inter-times
            valid_inter_times = valid_inter_times[valid_inter_times > 0]
            all_inter_times.append(valid_inter_times.cpu())

        # Concatenate all inter-times
        all_inter_times = torch.cat(all_inter_times, dim=0)
        logger.info(f"Collected {len(all_inter_times)} inter-time samples")

        # Create histogram using PyTorch
        min_time = torch.min(all_inter_times)
        max_time = torch.max(all_inter_times)

        # Create bin edges
        bin_edges = torch.linspace(min_time, max_time, self.num_bins + 1)

        # Calculate histogram
        counts = torch.histc(
            all_inter_times,
            bins=self.num_bins,
            min=min_time.item(),
            max=max_time.item(),
        )

        # Calculate bin centers
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate probabilities
        total_count = torch.sum(counts)
        self.bin_probabilities = counts / total_count

        # Store bin edges for reference
        self.bins = bin_edges

        logger.info(f"Built distribution with {self.num_bins} bins")
        logger.info(f"Inter-time range: [{min_time:.6f}, {max_time:.6f}]")

    def _sample_from_distribution(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Sample inter-times from the empirical distribution.

        Args:
            size: Shape of the tensor to generate (batch_size, seq_len)

        Returns:
            Tensor of sampled inter-times
        """
        total_samples = size[0] * size[1]

        # Sample bin indices according to probabilities using PyTorch
        bin_indices = torch.multinomial(
            self.bin_probabilities, num_samples=total_samples, replacement=True
        )

        # Get values from selected bins (use bin centers)
        sampled_values = self.bin_centers[bin_indices]

        # Add some noise within bins for better approximation
        bin_width = self.bins[1] - self.bins[0]  # Assuming uniform bin width
        noise = torch.rand(total_samples) * bin_width - bin_width / 2
        sampled_values = sampled_values + noise

        # Ensure positive values
        sampled_values = torch.clamp(sampled_values, min=1e-6)

        # Reshape
        sampled_tensor = sampled_values.reshape(size)

        return sampled_tensor

    def _create_time_predictions(self, batch: Tuple) -> torch.Tensor:
        """
        Create time predictions by sampling from the inter-time distribution.

        Args:
            batch: Input batch

        Returns:
            Tensor of predicted inter-times
        """
        time_delta_seqs = batch["time_delta_seqs"]
        batch_size, seq_len = time_delta_seqs.shape

        # Sample inter-times from distribution
        pred_inter_times = self._sample_from_distribution((batch_size, seq_len))

        # Move to same device as input
        if time_delta_seqs.device != pred_inter_times.device:
            pred_inter_times = pred_inter_times.to(time_delta_seqs.device)

        return pred_inter_times

    @property
    def benchmark_name(self) -> str:
        return "intertime_distribution_sampling"

    def _prepare_benchmark(self) -> None:
        self._build_intertime_distribution()

    def _get_custom_results_info(self) -> Dict[str, Any]:
        """Get custom information to add to results."""
        return {"num_bins": self.num_bins}
