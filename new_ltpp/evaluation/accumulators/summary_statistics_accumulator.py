"""
Batch Statistics Collector

Main orchestrator class that manages multiple accumulators and collects
statistics batch-by-batch during the prediction phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from new_ltpp.evaluation.statistical_testing.statistical_tests.builder import (
        StatisticalTestConfig,
    )

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
from .summary_stats.summary_stats_helper import SummaryStatsHelper
from .plot_generators import (
    AutocorrelationPlotGenerator,
    EventTypePlotGenerator,
    InterEventTimePlotGenerator,
    SequenceLengthPlotGenerator,
    StatTestPlotGenerator,
)
from .time_accumulator import InterEventTimeAccumulator
from .base_accumulator import IAccumulator, Accumulator
from .statistical_metrics_accumulator import StatisticalTestAccumulator


class AccumulatorContainer(TypedDict):
    time: InterEventTimeAccumulator
    event_type: EventTypeAccumulator
    sequence_length: SequenceLengthAccumulator
    correlation: CorrAccumulator
    statistical_tests: Optional[StatisticalTestAccumulator]


class BatchStatisticsCollector(Accumulator):
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
        output_dir: Path | str,
        statistical_test_config: Optional["StatisticalTestDict"] = None,
        dtime_min: float = 0.0,
        min_sim_events: int = 1,
        enable_plots: bool = True,
        enable_metrics: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the batch statistics collector.

        Args:
            num_event_types: Number of event types in the dataset
            dtime_max: Maximum inter-event time
            statistical_test_config: Configuration for statistical tests (e.g., MMD test)
            dtime_min: Minimum inter-event time
            min_sim_events: Minimum number of simulated events required per batch
            output_dir: Directory where results will be saved
            metadata: Additional metadata dictionary to log and save
        """
        self.num_event_types = num_event_types
        self.output_dir = Path(output_dir)
        # Minimum number of simulated events required per batch
        self.min_sim_events = int(min_sim_events)
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize base accumulators
        base_accumulators = AccumulatorContainer(
            time=InterEventTimeAccumulator(
                dtime_min=dtime_min, dtime_max=dtime_max, min_sim_events=min_sim_events
            ),
            event_type=EventTypeAccumulator(
                num_event_types=num_event_types, min_sim_events=min_sim_events
            ),
            sequence_length=SequenceLengthAccumulator(min_sim_events=min_sim_events),
            correlation=CorrAccumulator(min_sim_events=min_sim_events),
            statistical_tests=None,
        )

        if statistical_test_config is not None:
            # Override num_classes to ensure consistency with num_event_types
            statistical_test_config["num_classes"] = num_event_types
            base_accumulators.append(
                StatisticalTestAccumulator(
                    statistical_test_config, min_sim_events=min_sim_events
                )
            )

        self._accumulators: AccumulatorContainer = base_accumulators

        self._plot_generators = (
            InterEventTimePlotGenerator(),
            EventTypePlotGenerator(self.num_event_types),
            SequenceLengthPlotGenerator(),
            AutocorrelationPlotGenerator(),
            StatTestPlotGenerator(),
        )

        # Initialize metrics calculator (if enabled)
        self._summary_stats_helper: Optional[SummaryStatsHelper] = SummaryStatsHelper(
            num_event_types=self.num_event_types
        )

        self.metadata = metadata or {}
        if self.metadata:
            import json
            logger.info(f"Initialized Evaluation Phase with Metadata:\n{json.dumps(self.metadata, indent=2)}")

        # State tracking
        self._batch_count: int = 0
        self._is_finalized: bool = False

        logger.info(
            f"BatchStatisticsCollector initialized with {len(self._accumulators)} accumulators"
        )

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Update all accumulators with new batch data.

        This method should be called in the predict_step for each batch.

        Args:
            batch: Ground truth batch data
            simulation: Optional simulation results for the batch

        Returns:
        """

        # Update all accumulators (each validates simulation independently)
        for key, accumulator in self._accumulators.items():
            if accumulator is not None:
                cast(IAccumulator, accumulator).update(batch, simulation)

        self._batch_count += 1

        # Log progress periodically
        if self._batch_count % 100 == 0:
            for acc in self._accumulators.values():
                if acc is not None:
                    logger.info(
                        f"Batch {self._batch_count}: {type(acc).__name__} sample count = {cast(IAccumulator, acc).sample_count}"
                    )

        return

    def compute(self) -> AllStatistics:  # type: ignore[override]
        """Compute final statistics from all accumulators.

        Returns:
            Dictionary containing all computed statistics
        """
        logger.info("Computing statistics from accumulators...")

        # Get base statistics
        base_stats = AllStatistics(
            time=self._accumulators["time"].compute(),
            event_type=self._accumulators["event_type"].compute(),
            sequence_length=self._accumulators["sequence_length"].compute(),
            correlation=self._accumulators["correlation"].compute(),
            statistical_tests=(
                self._accumulators["statistical_tests"].compute()
                if self._accumulators["statistical_tests"] is not None
                else None
            ),
        )

        return base_stats

    def generate_plots(self, statistics: AllStatistics) -> None:
        """Generate and save all plots.

        Args:
            statistics: Dictionary containing computed statistics
        """
        logger.info("Generating comparison plots...")

        # Prepare data in the format expected by plot generators

        observed_statistic = (
            np.array(statistics["statistical_tests"]["observed_statistic"])
            if statistics["statistical_tests"] is not None
            else None
        )

        permuted_statistic = (
            np.array(statistics["statistical_tests"]["permuted_statistic"])
            if statistics["statistical_tests"] is not None
            else None
        )

        p_values = (
            np.array(statistics["statistical_tests"]["p_values"])
            if statistics["statistical_tests"] is not None
            else None
        )

        plot_data = PlotData(
            label_time_deltas=statistics["time"]["gt_time_deltas"],
            simulated_time_deltas=statistics["time"]["sim_time_deltas"],
            time_bin_edges=statistics["time"]["bin_edges"],
            label_event_types=statistics["event_type"]["gt_array"],
            simulated_event_types=statistics["event_type"]["sim_array"],
            label_sequence_lengths=statistics["sequence_length"]["gt_array"],
            simulated_sequence_lengths=statistics["sequence_length"]["sim_array"],
            acf_gt_mean=statistics["correlation"]["acf_gt_mean"],
            acf_sim_mean=statistics["correlation"]["acf_sim_mean"],
            observed_statistic=observed_statistic,
            permuted_statistic=permuted_statistic,
            p_values=p_values,
        )

        # Generate plots
        plot_filenames: List[str] = [
            "comparison_inter_event_time_dist.png",
            "comparison_event_type_dist.png",
            "comparison_sequence_length_dist.png",
            "comparison_autocorrelation.png",
            "mmd_test_distribution.png",
        ]

        for i, (generator, filename) in enumerate(
            zip(self._plot_generators, plot_filenames)
        ):
            output_path: str = str(self.output_dir / filename)

            generator.generate_plot(plot_data, output_path)
            logger.info(f"Generated plot: {filename}")

    def finalize_and_save(self, output_dir: Optional[Path | str] = "simulation_results") -> FinalResult:
        """Finalize collection, compute statistics, generate plots, and save results.

        This method should be called after all batches have been processed.

        Args:
            generate_plots: Whether to generate plots

        Returns:
            Dictionary containing:
                - 'statistics': All computed statistics
                - 'metrics': Summary metrics
                - 'batch_count': Number of batches processed

        Side effects:
            - Generates plots if generate_plots is True
            - Saves plots to output_dir
        """

        if output_dir is not None:
            output_dir = OUTPUT_DIR / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.output_dir

        import json

        if self._is_finalized:
            logger.warning("Collector already finalized, computing statistics anyway")
            # Continue to compute instead of returning empty result

        logger.info(
            f"Finalizing statistics collection after {self._batch_count} batches"
        )

        # Compute statistics
        statistics: AllStatistics = self.compute()

        # Compute metrics
        metrics: dict[str, float] = {}
        if self._summary_stats_helper is not None:
            metrics.update(self._summary_stats_helper.compute_metrics(statistics))

        # Add statistical test metrics
        if statistics.get("statistical_tests") is not None:
            stat_tests = statistics["statistical_tests"]
            if stat_tests and stat_tests.get("p_values"):
                metrics["statistical_test_mean_p_value"] = float(
                    np.mean(stat_tests["p_values"])
                )
            if stat_tests and stat_tests.get("observed_statistic"):
                metrics["statistical_test_mean_statistic"] = float(
                    np.mean(stat_tests["observed_statistic"])
                )

        # Generate plots
        if generate_plots:
            self.generate_plots(statistics)

        # Calculate metrics if enabled
        metrics_dict: Optional[Dict[str, float]] = None
        if self.enable_metrics:
            metrics_dict = {}
            
            # Add statistical test (MMD) metrics
            stat_tests = statistics.get("statistical_tests", {})
            if stat_tests and stat_tests.get("mmd_values"):
                import numpy as np
                metrics_dict["mean_mmd_value"] = float(np.mean(stat_tests["mmd_values"]))
                metrics_dict["std_mmd_value"] = float(np.std(stat_tests["mmd_values"]))
            
            if stat_tests and stat_tests.get("mmd_p_values"):
                import numpy as np
                metrics_dict["mean_mmd_p_value"] = float(np.mean(stat_tests["mmd_p_values"]))
                metrics_dict["std_mmd_p_value"] = float(np.std(stat_tests["mmd_p_values"]))
            
            # Save metrics to JSON
            metrics_path = output_dir / "metrics.json"
            try:
                with open(metrics_path, "w") as f:
                    json.dump(metrics_dict, f, indent=4)
                logger.info(f"Saved metrics to {metrics_path}")
            except Exception as e:
                logger.error(f"Failed to save metrics to {metrics_path}: {e}")

        # Save metadata to JSON
        if self.metadata:
            metadata_path = output_dir / "metadata.json"
            try:
                with open(metadata_path, "w") as f:
                    json.dump(self.metadata, f, indent=4)
                logger.info(f"Saved execution metadata to {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to save metadata to {metadata_path}: {e}")

        self._is_finalized = True

        result = FinalResult(
            statistics=statistics,
            metrics=metrics_dict,
            batch_count=self._batch_count,
        )

        logger.info("Statistics collection finalized successfully")
        return result

    def reset(self) -> None:
        """Reset all accumulators and state."""
        for accumulator in self._accumulators:
            cast(IAccumulator, accumulator).reset()
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
        return {
            type(acc).__name__: cast(IAccumulator, acc).sample_count
            for acc in self._accumulators
        }
