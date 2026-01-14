"""
Batch Statistics Collector

Main orchestrator class that manages multiple accumulators and collects
statistics batch-by-batch during the prediction phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from new_ltpp.globals import OUTPUT_DIR
from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .acc_types import (
    AllStatistics,
    FinalResult,
    PlotData,
)
from .corr_accumulator import CorrAccumulator
from .event_type_accumulator import EventTypeAccumulator
from .mean_len_accumulator import SequenceLengthAccumulator
from .metrics_calculator import MetricsCalculatorImpl
from .plot_generators import (
    AutocorrelationPlotGenerator,
    EventTypePlotGenerator,
    InterEventTimePlotGenerator,
    SequenceLengthPlotGenerator,
)
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
        )

        # In predict_step (called for each batch)
        collector.update_batch(batch, simulation)

        # After prediction loop (generate results)
        collector.finalize_and_save()
    """

    def __init__(
        self,
        num_event_types: int,
        dtime_max: float,
        dtime_min: float = 0.0,
        min_sim_events: int = 1,
        enable_plots: bool = True,
        enable_metrics: bool = True,
        output_dir: Path | str = OUTPUT_DIR / "distribution_comparison",
    ):
        """Initialize the batch statistics collector.

        Args:
            num_event_types: Number of event types in the dataset
            output_dir: Directory where results will be saved
            enable_plots: Whether to generate plots
            enable_metrics: Whether to compute metrics
        """
        self.num_event_types = num_event_types
        self.output_dir = Path(output_dir)
        # Minimum number of simulated events required per batch
        self.min_sim_events = int(min_sim_events)
        self.enable_plots = enable_plots
        self.enable_metrics = enable_metrics

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accumulators
        self._accumulators = (
            InterEventTimeAccumulator(
                dtime_min=dtime_min, dtime_max=dtime_max, min_sim_events=min_sim_events
            ),
            EventTypeAccumulator(
                num_event_types=num_event_types, min_sim_events=min_sim_events
            ),
            SequenceLengthAccumulator(min_sim_events=min_sim_events),
            CorrAccumulator(min_sim_events=min_sim_events),
        )

        # Initialize plot generators (if enabled)
        self._plot_generators: Optional[
            Tuple[
                InterEventTimePlotGenerator,
                EventTypePlotGenerator,
                SequenceLengthPlotGenerator,
                AutocorrelationPlotGenerator,
            ]
        ] = None

        if self.enable_plots:
            self._plot_generators = (
                InterEventTimePlotGenerator(),
                EventTypePlotGenerator(self.num_event_types),
                SequenceLengthPlotGenerator(),
                AutocorrelationPlotGenerator(),
            )

        # Initialize metrics calculator (if enabled)
        self._metrics_calculator: Optional[MetricsCalculatorImpl] = (
            MetricsCalculatorImpl() if self.enable_metrics else None
        )

        # State tracking
        self._batch_count: int = 0
        self._is_finalized: bool = False

        logger.info(
            f"BatchStatisticsCollector initialized with {len(self._accumulators)} accumulators"
        )

    def update_batch(self, batch: Batch, simulation: SimulationResult) -> bool:
        """Update all accumulators with new batch data.

        This method should be called in the predict_step for each batch.

        Args:
            batch: Ground truth batch data
            simulation: Optional simulation results for the batch

        Returns:
        """

        if self._is_finalized:
            logger.warning("Collector already finalized, ignoring update")
            return False

        # Update all accumulators (each validates simulation independently)
        for accumulator in self._accumulators:
            accumulator.update(batch, simulation)

        self._batch_count += 1

        # Log progress periodically
        if self._batch_count % 100 == 0:
            sample_counts = {
                type(acc).__name__: acc.sample_count for acc in self._accumulators
            }
            logger.info(
                f"Processed {self._batch_count} batches. Sample counts: {sample_counts}"
            )

        return True

    def compute_statistics(self) -> AllStatistics:
        """Compute final statistics from all accumulators.

        Returns:
            Dictionary containing all computed statistics
        """
        logger.info("Computing statistics from accumulators...")

        return AllStatistics(
            time=self._accumulators[0].compute(),
            event_type=self._accumulators[1].compute(),
            sequence_length=self._accumulators[2].compute(),
            correlation=self._accumulators[3].compute(),
        )

    def generate_plots(self, statistics: AllStatistics) -> None:
        """Generate and save all plots.

        Args:
            statistics: Dictionary containing computed statistics
        """
        if not self.enable_plots or self._plot_generators is None:
            logger.info("Plot generation disabled")
            return

        logger.info("Generating comparison plots...")

        # Prepare data in the format expected by plot generators
        plot_data = PlotData(
            label_time_deltas=statistics["time"]["gt_time_deltas"],
            simulated_time_deltas=statistics["time"]["sim_time_deltas"],
            time_bin_edges=statistics["time"]["bin_edges"],
            label_event_types=statistics["event_type"]["gt_array"],
            simulated_event_types=statistics["event_type"]["sim_array"],
            label_sequence_lengths=statistics["sequence_length"]["gt_array"],
            simulated_sequence_lengths=statistics["sequence_length"]["sim_array"],
        )

        # Prepare ACF data separately (not in PlotData TypedDict)
        acf_data = {
            "acf_gt_mean": statistics["correlation"]["acf_gt_mean"],
            "acf_sim_mean": statistics["correlation"]["acf_sim_mean"],
        }

        # Generate plots
        plot_filenames: List[str] = [
            "comparison_inter_event_time_dist.png",
            "comparison_event_type_dist.png",
            "comparison_sequence_length_dist.png",
            "comparison_autocorrelation.png",
        ]

        # Type assertion safe because we checked self._plot_generators is not None
        assert self._plot_generators is not None
        # Cast TypedDict to dict for plot generators
        plot_data_dict: Dict[str, Any] = dict(plot_data)
        for i, (generator, filename) in enumerate(
            zip(self._plot_generators, plot_filenames)
        ):
            output_path: str = str(self.output_dir / filename)
            # Use acf_data for the last plot (autocorrelation)
            data_to_use = acf_data if i == 3 else plot_data_dict
            generator.generate_plot(data_to_use, output_path)
            logger.info(f"Generated plot: {filename}")

    def finalize_and_save(self) -> FinalResult:
        """Finalize collection, compute statistics, generate plots, and save results.

        This method should be called after all batches have been processed.

        Returns:
            Dictionary containing:
                - 'statistics': All computed statistics
                - 'metrics': Summary metrics
                - 'batch_count': Number of batches processed
        """
        if self._is_finalized:
            logger.warning("Collector already finalized, computing statistics anyway")
            # Continue to compute instead of returning empty result

        logger.info(
            f"Finalizing statistics collection after {self._batch_count} batches"
        )

        # Compute statistics
        statistics: AllStatistics = self.compute_statistics()

        # Generate plots
        self.generate_plots(statistics)

        self._is_finalized = True

        result = FinalResult(
            statistics=statistics,
            batch_count=self._batch_count,
        )

        logger.info("Statistics collection finalized successfully")
        return result

    def reset(self) -> None:
        """Reset all accumulators and state."""
        for accumulator in self._accumulators:
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
        return {type(acc).__name__: acc.sample_count for acc in self._accumulators}
