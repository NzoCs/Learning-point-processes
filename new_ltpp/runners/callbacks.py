import pytorch_lightning as pl
from typing import cast
import pickle
from new_ltpp.models.model_protocol import ITPPModel
from new_ltpp.evaluation.accumulators.summary_statistics_accumulator import FinalResult


def save_final_results(final_results: FinalResult, save_dir: str):
    """Save the final results to a file."""
    with open(save_dir + "/final_results.pkl", "wb") as f:
        pickle.dump(final_results, f)


class PredictionStatsCallback(pl.Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called once before prediction begins."""
        model = cast(ITPPModel, pl_module)
        model.init_statistics_collector(output_dir=self.output_dir)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called once after prediction ends."""
        model = cast(ITPPModel, pl_module)
        if model._statistics_collector is None:
            raise NotImplementedError(
                "No statistics collector to finalize. Initialize it first with 'init_statistics_collector'."
            )
        final_results = model._statistics_collector.finalize(generate_plots=True)
        save_final_results(final_results, self.output_dir)
        model.intensity_graph(save_dir=self.output_dir)


class TestCallback(pl.Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called once before test begins."""
        model = cast(ITPPModel, pl_module)
        model.init_statistics_collector(output_dir=self.output_dir)
