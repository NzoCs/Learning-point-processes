from typing import Any

import pytorch_lightning as pl

from new_ltpp.configs import RunnerConfig
from new_ltpp.configs.logger_config import LoggerFactory
from new_ltpp.data.preprocess import TPPDataModule
from new_ltpp.models.model_factory import ModelFactory
from new_ltpp.runners.trainer_factory import (
    CheckpointManager,
    PredictionStatsCallback,
    TrainerFactory,
)
from new_ltpp.utils import logger


class Runner:

    def __init__(
        self,
        config: RunnerConfig,
        enable_logging: bool = True,
    ):
        """_summary__.
        Args:
            config (RunnerConfig): Configuration object containing all the necessary parameters for training.
            checkpoint_path (str, optional): Path to a checkpoint file to resume training from. Defaults to None.
            **kwargs: Additional keyword arguments that can be used to override specific configurations.
        """
        # Initialize your configs

        # Initialize your datamodule
        self.datamodule = TPPDataModule(config.data_config)
        data_info = self.datamodule.get_data_info()

        # Initialize your model
        # Use the ModelFactory to create the model
        self.model = ModelFactory.create_model_by_name(
            model_name=config.model_id,
            model_config=config.model_config,
            data_info=data_info,
            output_dir=config.base_dir,
        )

        self.model_id = config.model_id

        self.dirpath = config.checkpoints_dir
        self.logger_config = config.logger_config
        config.logger_config.save_dir = str(
            self.dirpath / config.logger_config.save_dir
        )

        self.checkpoint_path = CheckpointManager(str(self.dirpath)).latest_best()

        self.dataset_id = config.data_config.dataset_id
        self.enable_logging = enable_logging
        self._lightning_logger = self._build_lightning_logger(enable_logging)
        self.config = config  # Store config for access in save method

    def set_logging(self, enable_logging: bool):
        """Toggle logging, causing trainer recreation if needed."""
        if self.enable_logging == enable_logging:
            return

        self.enable_logging = enable_logging
        self._lightning_logger = self._build_lightning_logger(enable_logging)

    def _build_lightning_logger(self, enable_logging: bool) -> Any | None:
        if not enable_logging:
            return None
        return LoggerFactory.create_logger(self.logger_config)

    @property
    def trainer(self) -> pl.Trainer:
        """Create trainer with all necessary callbacks for the current phase."""
        trainer = TrainerFactory.create(
            training_config=self.config.training_config,
            trainer_logger=self._lightning_logger,
            checkpoints_dir=self.dirpath,
            extra_callbacks=[
                PredictionStatsCallback(
                    output_dir=str(self.config.base_dir / "distribution_comparison")
                )
            ],
        )
        return trainer

    def train(self) -> None:
        """Training a model with optional resumption from a checkpoint."""

        logger.info(
            f"--- Starting Training for Model : {self.model_id} on dataset : {self.dataset_id} ---"
        )

        trainer = self.trainer
        # Train the model
        self.datamodule.setup(stage="fit")

        train_dataloader = self.datamodule.train_dataloader()
        val_dataloader = self.datamodule.val_dataloader()

        trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=self.checkpoint_path,
        )

    def test(self) -> None:
        """
        Test the model with optional custom parameters for the test_step method.
        Results are saved to a JSON file in the model directory.
        """

        logger.info(
            f"--- Starting Testing for Model : {self.model_id} on dataset : {self.dataset_id} ---"
        )

        trainer = self.trainer
        self.datamodule.setup(stage="test")

        test_dataloader = self.datamodule.test_dataloader()

        results = trainer.test(
            model=self.model,
            dataloaders=test_dataloader,
            ckpt_path=self.checkpoint_path,
        )

        # Save test results
        if results and len(results) > 0:
            import json

            test_results_dir = self.config.base_dir / "test_results"
            test_results_dir.mkdir(parents=True, exist_ok=True)
            results_file = test_results_dir / "test_results.json"

            with open(results_file, "w") as f:
                json.dump(results[0], f, indent=4)

            logger.info(f"Test results saved to {results_file}")

    def predict(self) -> None:
        """
        Run predictions (e.g., simulations) using the model and save results.

        The PredictionStatsCallback handles statistics finalization and intensity
        graph generation automatically while the model is still on the correct device.
        """
        logger.info(
            f"--- Starting Prediction for Model : {self.model_id} on dataset : {self.dataset_id} ---"
        )

        trainer = self.trainer
        self.datamodule.setup(stage="predict")

        predict_dataloader = self.datamodule.test_dataloader()

        # The callback will handle finalize_statistics() and intensity_graph()
        trainer.predict(
            model=self.model,
            dataloaders=predict_dataloader,
            ckpt_path=self.checkpoint_path,
        )
