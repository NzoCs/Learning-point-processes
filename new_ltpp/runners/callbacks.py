"""Lightning callbacks for simulation and statistics collection."""

import pytorch_lightning as pl
from typing import cast
from pathlib import Path

from new_ltpp.configs.statistical_test_config import (
    SimulationConfig,
    StatisticalTestConfig,
)
from new_ltpp.globals import OUTPUT_DIR
from new_ltpp.models.model_protocol import ISimulableModel
from new_ltpp.models.simulation.simulator import Simulator
from new_ltpp.models.simulation.tpp_io import SimulationIOManager
from new_ltpp.models.visualization.model_visualizer import ModelVisualizer
from new_ltpp.evaluation.results_aggregator import ResultsAggregator


def _get_simulator(pl_module: pl.LightningModule) -> "Simulator":
    """Extract Simulator from a Model instance."""
    if not hasattr(pl_module, "_simulator"):
        raise AttributeError(
            f"{pl_module.__class__.__name__} has no '_simulator' attribute. "
            "Ensure it inherits from base_model.Model."
        )

    simulator = cast(Simulator, pl_module._simulator)
    return simulator


def _get_visualizer(pl_module: pl.LightningModule) -> "ModelVisualizer":
    """Extract ModelVisualizer from a Model instance."""
    if not hasattr(pl_module, "_visualizer"):
        raise AttributeError(
            f"{pl_module.__class__.__name__} has no '_visualizer' attribute. "
            "Ensure it inherits from base_model.Model."
        )
    visualizer = cast(ModelVisualizer, pl_module._visualizer)
    return visualizer


class PredictionStatsCallback(pl.Callback):
    """Initializes and finalizes the statistics collector around the predict loop.

    Hooks:
        on_predict_start → simulator.init_statistics_collector(output_dir)
        on_predict_end   → statistics_collector.finalize_and_save() + visualizer.intensity_graph()
    """

    def __init__(
        self,
        base_dir: Path,
        statistical_test_config: "StatisticalTestConfig",
        simulation_config: "SimulationConfig",
        metadata: dict | None = None,
        experiment_id: str | None = None,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.statistical_test_config = statistical_test_config
        self.simulation_config = simulation_config
        self.metadata = metadata or {}
        self.experiment_id = experiment_id or base_dir.name
        self.aggregator = ResultsAggregator(csv_path=OUTPUT_DIR / "global_results.csv")

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        model = cast("ISimulableModel", pl_module)
        simulator = Simulator(
            model=model,
            statistical_test_config=self.statistical_test_config,
        )
        model._simulator = simulator

        # Initialize and inject SimulationIOManager
        io_manager = SimulationIOManager(num_event_types=model.num_event_types)
        io_manager.setup_io(output_dir=self.base_dir / "simulations")
        model._io_manager = io_manager

        simulator.init_statistics_collector(base_dir=self.base_dir)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        model = cast("ISimulableModel", pl_module)
        simulator = model._simulator
        if simulator._statistics_collector is None:
            raise RuntimeError(
                "Statistics collector not initialized. Check on_predict_start implementation."
            )
        sim_metrics = simulator._statistics_collector.finalize_and_save(
            generate_plots=True
        )

        # Aggregate results
        self.aggregator.add_result(
            experiment_id=self.experiment_id,
            metadata=self.metadata,
            sim_metrics=sim_metrics,
        )

        # Finalize Simulation IO (Close the single parquet file)
        if model._io_manager is not None:
            model._io_manager.finalize()

        visualizer = ModelVisualizer(model)
        visualizer.intensity_graph(
            save_dir=self.base_dir / "intensities",
            save_plot=True,
            save_data=True,
            plot=False,
        )


class TestCallback(pl.Callback):
    """Initializes the statistics collector before the test loop.

    Hooks:
    """

    def __init__(
        self,
        output_dir: str | Path,
        metadata: dict | None = None,
        experiment_id: str | None = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.metadata = metadata or {}
        self.experiment_id = experiment_id or self.output_dir.parent.name
        self.aggregator = ResultsAggregator(csv_path=OUTPUT_DIR / "global_results.csv")

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Aggregate test metrics into global results."""
        # Extract all metrics from trainer, ignoring system ones
        exclude = ["v_num", "epoch", "step"]
        test_metrics = {
            k: v.item() if hasattr(v, "item") else v
            for k, v in trainer.callback_metrics.items()
            if k not in exclude
        }

        self.aggregator.add_result(
            experiment_id=self.experiment_id,
            metadata=self.metadata,
            test_metrics=test_metrics,
        )
