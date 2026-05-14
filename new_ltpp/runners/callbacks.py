import pytorch_lightning as pl
from typing import cast
from new_ltpp.models.model_protocol import ITPPModel


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
        model._statistics_collector.finalize_and_save(generate_plots=True)
        model.intensity_graph(save_dir=self.output_dir)


class TestCallback(pl.Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called once before test begins."""
        model = cast(ITPPModel, pl_module)
        model.init_statistics_collector(output_dir=self.output_dir)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass
        """Called once after test ends. doit enregistrer les resultats du test dans un fichier pickle pour les analyser plus tard."""
