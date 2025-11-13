"""
Batch Statistics Collector

Main orchestrator class that manages multiple accumulators and collects
statistics batch-by-batch during the prediction phase.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .base_accumulator import BaseAccumulator
from .event_type_accumulator import EventTypeAccumulator
from .metrics_calculator import MetricsCalculatorImpl
from .moment_accumulator import MomentAccumulator
from .plot_generators import (
    CrossCorrelationPlotGenerator,
    EventTypePlotGenerator,
    InterEventTimePlotGenerator,
    SequenceLengthPlotGenerator,
)
from .sequence_length_accumulator import SequenceLengthAccumulator
from .time_accumulator import InterEventTimeAccumulator


class BatchStatisticsCollector:
    """Main class for collecting statistics batch-by-batch during prediction.
    
    This class orchestrates multiple accumulators that collect different
    statistical properties from batches of ground truth and simulated data.
    It's designed to be called in the predict_step of a model.
    
    Usage:
        # Initialization (once, before prediction loop)
        collector = BatchStatisticsCollector(
            num_event_types=10,
            output_dir="./output",
            max_samples=100000
        )
        
        # In predict_step (called for each batch)
        collector.update_batch(batch, simulation)
        
        # After prediction loop (generate results)
        collector.finalize_and_save()
    """

    def __init__(
        self,
        num_event_types: int,
        output_dir: str,
        max_samples: Optional[int] = None,
        enable_plots: bool = True,
        enable_metrics: bool = True,
    ):
        """Initialize the batch statistics collector.
        
        Args:
            num_event_types: Number of event types in the dataset
            output_dir: Directory where results will be saved
            max_samples: Maximum number of samples to collect (None for unlimited)
            enable_plots: Whether to generate plots
            enable_metrics: Whether to compute metrics
        """
        self.num_event_types = num_event_types
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.enable_plots = enable_plots
        self.enable_metrics = enable_metrics

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accumulators
        self._accumulators: Dict[str, BaseAccumulator] = {
            'time': InterEventTimeAccumulator(max_samples=max_samples),
            'event_type': EventTypeAccumulator(num_event_types=num_event_types, max_samples=max_samples),
            'sequence_length': SequenceLengthAccumulator(max_samples=max_samples),
            'moments': MomentAccumulator(max_samples=max_samples),
        }

        # Initialize plot generators (if enabled)
        self._plot_generators = []
        if self.enable_plots:
            self._plot_generators = [
                InterEventTimePlotGenerator(),
                EventTypePlotGenerator(num_event_types),
                SequenceLengthPlotGenerator(),
                CrossCorrelationPlotGenerator(),
            ]

        # Initialize metrics calculator (if enabled)
        self._metrics_calculator = MetricsCalculatorImpl() if self.enable_metrics else None

        # State tracking
        self._batch_count = 0
        self._is_finalized = False

        logger.info(f"BatchStatisticsCollector initialized with {len(self._accumulators)} accumulators")

    def update_batch(self, batch: Batch, simulation: Optional[SimulationResult] = None) -> bool:
        """Update all accumulators with new batch data.
        
        This method should be called in the predict_step for each batch.
        
        Args:
            batch: Ground truth batch data
            simulation: Optional simulation results for the batch
            
        Returns:
            bool: True if collection should continue, False if max_samples reached
        """
        if self._is_finalized:
            logger.warning("Collector already finalized, ignoring update")
            return False

        # Check if we should continue collecting
        should_continue = any(acc.should_continue() for acc in self._accumulators.values())
        if not should_continue:
            logger.info(f"Max samples reached across all accumulators after {self._batch_count} batches")
            return False

        # Update all accumulators
        for name, accumulator in self._accumulators.items():
            try:
                accumulator.update(batch, simulation)
            except Exception as e:
                logger.error(f"Error updating {name} accumulator: {str(e)}")

        self._batch_count += 1

        # Log progress periodically
        if self._batch_count % 100 == 0:
            sample_counts = {name: acc.sample_count for name, acc in self._accumulators.items()}
            logger.info(f"Processed {self._batch_count} batches. Sample counts: {sample_counts}")

        return True

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute final statistics from all accumulators.
        
        Returns:
            Dictionary containing all computed statistics
        """
        logger.info("Computing statistics from accumulators...")

        all_stats = {}
        for name, accumulator in self._accumulators.items():
            try:
                stats = accumulator.compute()
                all_stats[name] = stats
            except Exception as e:
                logger.error(f"Error computing statistics for {name}: {str(e)}")
                all_stats[name] = {}

        return all_stats

    def generate_plots(self, statistics: Dict[str, Any]) -> None:
        """Generate and save all plots.
        
        Args:
            statistics: Dictionary containing computed statistics
        """
        if not self.enable_plots:
            logger.info("Plot generation disabled")
            return

        logger.info("Generating comparison plots...")

        # Prepare data in the format expected by plot generators
        plot_data = {
            'label_time_deltas': statistics.get('time', {}).get('gt_time_deltas', []),
            'simulated_time_deltas': statistics.get('time', {}).get('sim_time_deltas', []),
            'label_event_types': statistics.get('event_type', {}).get('gt_event_types', []),
            'simulated_event_types': statistics.get('event_type', {}).get('sim_event_types', []),
            'label_sequence_lengths': statistics.get('sequence_length', {}).get('gt_seq_lengths', []),
            'simulated_sequence_lengths': statistics.get('sequence_length', {}).get('sim_seq_lengths', []),
        }

        # Generate plots
        plot_filenames = [
            "comparison_inter_event_time_dist.png",
            "comparison_event_type_dist.png",
            "comparison_sequence_length_dist.png",
            "comparison_cross_correlation_moments.png",
        ]

        for generator, filename in zip(self._plot_generators, plot_filenames):
            output_path = str(self.output_dir / filename)
            try:
                generator.generate_plot(plot_data, output_path)
                logger.info(f"Generated plot: {filename}")
            except Exception as e:
                logger.error(f"Error generating plot {filename}: {str(e)}")

    def compute_metrics(self, statistics: Dict[str, Any]) -> Dict[str, float]:
        """Compute summary metrics from statistics.
        
        Args:
            statistics: Dictionary containing computed statistics
            
        Returns:
            Dictionary of computed metrics
        """
        if not self.enable_metrics or self._metrics_calculator is None:
            logger.info("Metrics computation disabled")
            return {}

        logger.info("Computing summary metrics...")

        # Prepare data in the format expected by metrics calculator
        metrics_data = {
            'label_time_deltas': statistics.get('time', {}).get('gt_time_deltas', []),
            'simulated_time_deltas': statistics.get('time', {}).get('sim_time_deltas', []),
            'label_sequence_lengths': statistics.get('sequence_length', {}).get('gt_seq_lengths', []),
            'simulated_sequence_lengths': statistics.get('sequence_length', {}).get('sim_seq_lengths', []),
        }

        try:
            metrics = self._metrics_calculator.calculate_metrics(metrics_data)
            
            # Add moment statistics
            moment_stats = statistics.get('moments', {})
            metrics.update({
                'gt_inter_event_time_mean': moment_stats.get('gt_mean', 0.0),
                'gt_inter_event_time_std': moment_stats.get('gt_std', 0.0),
                'gt_inter_event_time_skewness': moment_stats.get('gt_skewness', 0.0),
                'gt_inter_event_time_kurtosis': moment_stats.get('gt_kurtosis', 0.0),
                'sim_inter_event_time_mean': moment_stats.get('sim_mean', 0.0),
                'sim_inter_event_time_std': moment_stats.get('sim_std', 0.0),
                'sim_inter_event_time_skewness': moment_stats.get('sim_skewness', 0.0),
                'sim_inter_event_time_kurtosis': moment_stats.get('sim_kurtosis', 0.0),
            })
            
            return metrics
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {}

    def finalize_and_save(self) -> Dict[str, Any]:
        """Finalize collection, compute statistics, generate plots, and save results.
        
        This method should be called after all batches have been processed.
        
        Returns:
            Dictionary containing:
                - 'statistics': All computed statistics
                - 'metrics': Summary metrics
                - 'batch_count': Number of batches processed
        """
        if self._is_finalized:
            logger.warning("Collector already finalized")
            return {}

        logger.info(f"Finalizing statistics collection after {self._batch_count} batches")

        # Compute statistics
        statistics = self.compute_statistics()

        # Generate plots
        self.generate_plots(statistics)

        # Compute metrics
        metrics = self.compute_metrics(statistics)

        # Save metrics to file
        if metrics:
            metrics_file = self.output_dir / "distribution_metrics.txt"
            try:
                with open(metrics_file, 'w') as f:
                    f.write("Distribution Comparison Metrics\n")
                    f.write("=" * 50 + "\n\n")
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                logger.info(f"Metrics saved to {metrics_file}")
            except Exception as e:
                logger.error(f"Error saving metrics: {str(e)}")

        self._is_finalized = True

        result = {
            'statistics': statistics,
            'metrics': metrics,
            'batch_count': self._batch_count,
        }

        logger.info("Statistics collection finalized successfully")
        return result

    def reset(self) -> None:
        """Reset all accumulators and state."""
        for accumulator in self._accumulators.values():
            accumulator.reset()
        self._batch_count = 0
        self._is_finalized = False
        logger.info("BatchStatisticsCollector reset")

    @property
    def batch_count(self) -> int:
        """Return number of batches processed."""
        return self._batch_count

    @property
    def sample_counts(self) -> Dict[str, int]:
        """Return sample counts for all accumulators."""
        return {name: acc.sample_count for name, acc in self._accumulators.items()}
