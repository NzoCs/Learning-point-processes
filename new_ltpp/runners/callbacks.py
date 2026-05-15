"""Lightning callbacks for simulation and statistics collection."""

import pytorch_lightning as pl
from typing import cast
from pathlib import Path

from new_ltpp.configs.statistical_test_config import (
    SimulationConfig,
    StatisticalTestConfig,
)
from new_ltpp.models.model_protocol import ISimulableModel
from new_ltpp.simulation.simulator import Simulator
from new_ltpp.visualization.model_visualizer import ModelVisualizer


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
    ):
        super().__init__()
        self.base_dir = base_dir
        self.statistical_test_config = statistical_test_config
        self.simulation_config = simulation_config

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        model = cast("ISimulableModel", pl_module)
        simulator = Simulator(
            model=model,
            statistical_test_config=self.statistical_test_config,
        )
        model._simulator = simulator
        simulator.init_statistics_collector(base_dir=self.base_dir)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        model = cast("ISimulableModel", pl_module)
        simulator = model._simulator
        if simulator._statistics_collector is None:
            raise RuntimeError(
                "Statistics collector not initialized. Check on_predict_start implementation."
            )
        simulator._statistics_collector.finalize_and_save(generate_plots=True)

        visualizer = ModelVisualizer(model)
        visualizer.intensity_graph(save_dir=self.base_dir / "distributions")


class TestCallback(pl.Callback):
    """Initializes the statistics collector before the test loop.

    Hooks:
    """

    def __init__(self, output_dir: str | Path):
        super().__init__()
        self.output_dir = output_dir

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Reserved: save test results to pickle for later analysis."""
        pass
