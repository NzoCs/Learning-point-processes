"""
Sample Distribution Mark Benchmark

This benchmark creates bins to approximate the distribution of event marks (types)
from the training dataset, then predicts marks by sampling from this distribution.
"""

from typing import Any, Dict, Tuple, Union
from pathlib import Path

import torch

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.shared_types import Batch
from new_ltpp.utils import logger
from new_ltpp.globals import OUTPUT_DIR

from .type_bench import TypeBenchmark


class MarkDistributionBenchmark(TypeBenchmark):
    """
    Benchmark that samples event marks from the empirical distribution of training data.
    """

    def __init__(self, data_config: DataConfig, save_dir: Union[str, Path] = OUTPUT_DIR / "benchmarks"):
        """
        Initialize the mark distribution benchmark.

        Args:
            data_config: Data configuration object
            save_dir: Directory to save results
        """
        super().__init__(data_config, save_dir)

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
            type_seqs = batch.type_seqs  # Event types/marks
            batch_non_pad_mask = batch.seq_non_pad_mask

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

    def _create_type_predictions(self, batch: Batch) -> torch.Tensor:
        """
        Create type predictions by sampling from the mark distribution.

        Args:
            batch: Input batch

        Returns:
            Tensor of predicted types
        """
        type_seqs = batch.type_seqs
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
