import os
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from easy_tpp.configs import RunnerConfig
from easy_tpp.configs.logger_config import LoggerFactory
from easy_tpp.data.preprocess import TPPDataModule
from easy_tpp.evaluation.distribution_analysis_helper import (
    NTPPComparatorFactory,
)
from easy_tpp.models.model_factory import ModelFactory
from easy_tpp.utils import logger


class Runner:

    def __init__(
        self,
        config: RunnerConfig,
        enable_logging: bool = True,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """_summary__.
        Args:
            config (RunnerConfig): Configuration object containing all the necessary parameters for training.
            checkpoint_path (str, optional): Path to a checkpoint file to resume training from. Defaults to None.
            **kwargs: Additional keyword arguments that can be used to override specific configurations.
        """
        # Set matmul precision for Tensor Cores (if available)
        # Recommended for A100 GPUs as per logs
        if torch.cuda.is_available():
            try:
                # Try to get device capability safely
                device_capability = torch.cuda.get_device_capability(0)
                if device_capability[0] >= 7:
                    # Check if the major capability is >= 7 (Volta, Ampere, Hopper, etc.)
                    torch.set_float32_matmul_precision("medium")
                    logger.info(
                        f"Set torch.set_float32_matmul_precision('medium') for Tensor Cores. GPU capability: {device_capability}"
                    )
                else:
                    logger.info(
                        f"GPU capability {device_capability} < 7.0, keeping default matmul precision."
                    )
            except RuntimeError as e:
                logger.warning(
                    f"Could not access GPU device capability (driver issue?): {e}"
                )
                logger.info("Continuing without Tensor Core optimization.")
            except Exception as e:
                logger.warning(f"Could not set matmul precision: {e}")

        # Initialize your configs
        data_config = config.data_config
        model_config = config.model_config
        training_config = config.training_config

        # Initialize your model

        # Utiliser la ModelFactory pour créer le modèle
        model_factory = ModelFactory()
        self.model = model_factory.create_model_by_name(
            model_name=config.model_id,
            num_event_types=data_config.tokenizer_specs.num_event_types,
            model_config=model_config
        )

        self.model_id = config.model_id

        # Initialize Dataloaders
        self.datamodule = TPPDataModule(data_config)

        # Initialize Train params
        self.max_epochs = training_config.max_epochs
        self.log_freq = training_config.log_freq
        self.checkpoints_freq = training_config.checkpoints_freq
        self.patience = training_config.patience
        self.devices = training_config.devices
        self.val_freq = training_config.val_freq
        self.use_precision_16 = training_config.use_precision_16
        self.accumulate_grad_batches = training_config.accumulate_grad_batches

        # Model saving directory
        self.dirpath = config.save_model_dir
        self.logger_config = config.logger_config


        if checkpoint_path is None:
            # Liste des checkpoints à tester, par ordre de priorité
            possible_checkpoints = [
                os.path.join(self.dirpath, f"best-v{10-i}.ckpt") for i in range(10)
            ] + [  # Celui donné en argument
                os.path.join(self.dirpath, "best.ckpt"),
                os.path.join(self.dirpath, "last.ckpt"),
            ]

            # Trouve le premier fichier existant dans la liste
            self.checkpoint_path_ = None
            for path in possible_checkpoints:
                if os.path.exists(path):
                    self.checkpoint_path_ = path
                    logger.info(f"Checkpoint found: loading from {path}")
                    break

        elif isinstance(checkpoint_path, str):
            checkpoint_path = checkpoint_path + ".ckpt"
            # Store checkpoint path for resuming training
            self.checkpoint_path_ = os.path.join(self.dirpath, checkpoint_path)
        else:
            raise ValueError("Checkpoint path must be a string or None.")

        if self.checkpoint_path_ and os.path.exists(self.checkpoint_path_):
            logger.info(f"Loading model from checkpoint: {self.checkpoint_path}.")
        else:
            logger.info("No valid checkpoint found. Starting from scratch.")

        self.dataset_id = data_config.dataset_id
        self.enable_logging = enable_logging
        self._cached_trainer = None  # Cache for the trainer

        self._configure_logging(enable_logging)

    @property
    def checkpoint_path(self):
        """Return the checkpoint path for resuming training."""
        if isinstance(self.checkpoint_path_, str) and os.path.exists(
            self.checkpoint_path_
        ):
            return self.checkpoint_path_
        return None

    def _configure_logging(self, enable_logging: bool = True):
        """Configure logging for the trainer."""
        if enable_logging:
            try:
                self.logger = LoggerFactory.create_logger(self.logger_config)
            except Exception as e:
                self.logger = False
                logger.critical(f"Logging is disabled for this run. Error: {str(e)}")
        else:
            self.logger = False

    def set_logging(self, enable_logging: bool):
        """Change logging configuration without recreating the trainer."""
        if self.enable_logging == enable_logging:
            return  # No change needed

        self.enable_logging = enable_logging
        self._configure_logging(enable_logging)

        # If we have a cached trainer, update its logger
        if hasattr(self, "_cached_trainer") and self._cached_trainer is not None:
            self._cached_trainer.logger = self.logger

    @property
    def callbacks(self):

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.dirpath,
            filename="best",
            save_top_k=1,
            mode="min",
            every_n_epochs=self.checkpoints_freq,
            auto_insert_metric_name=False,
            save_last=True,  # Save the last checkpoint as well
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            verbose=True,  # Increase verbosity to know when early stopping occurs
            mode="min",
        )

        return [checkpoint_callback, early_stop_callback]

    @property
    def trainer(self) -> pl.Trainer:
        # Cache the trainer to avoid recreating it
        if hasattr(self, "_cached_trainer") and self._cached_trainer is not None:
            return self._cached_trainer
        # If devices is a number > 0, use GPU(s) if available
        # If devices is "auto", let PyTorch Lightning decide
        if (
            isinstance(self.devices, int)
            and self.devices > 0
            and torch.cuda.is_available()
        ):
            devices = self.devices
            accelerator = "gpu"
            strategy = "ddp" if self.devices > 1 else "auto"
        else:
            devices = "auto"  # Could be "auto" or another value
            accelerator = "auto"
            strategy = "auto"

        # Whether to use float16 or float32
        if self.use_precision_16:
            precision = "16-mixed"
        else:
            precision = "32-true"

        # Check if distributed training is requested
        if self.devices > 1:
            # Use DDPStrategy with find_unused_parameters=True
            strategy = DDPStrategy(find_unused_parameters=True)
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                devices=self.devices,
                accelerator="gpu",
                strategy=strategy,
                logger=self.logger,
                log_every_n_steps=self.log_freq,
                callbacks=self.callbacks,
                enable_progress_bar=True,
                enable_model_summary=True,
                check_val_every_n_epoch=self.val_freq,
                precision="16-mixed" if self.use_precision_16 else "32-true",
                accumulate_grad_batches=self.accumulate_grad_batches,
            )

        else:
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                devices=devices,
                accelerator=accelerator,
                strategy=strategy,
                logger=self.logger,
                log_every_n_steps=self.log_freq,
                callbacks=self.callbacks,
                enable_progress_bar=True,
                enable_model_summary=True,
                check_val_every_n_epoch=self.val_freq,
                precision=precision,
                accumulate_grad_batches=self.accumulate_grad_batches,
            )

        # Cache the trainer
        self._cached_trainer = trainer
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

            results_file = os.path.join(self.dirpath, "test_results.json")

            with open(results_file, "w") as f:
                json.dump(results[0], f, indent=4)

            logger.info(f"Test results saved to {results_file}")

    def predict(self) -> None:
        """
        Run predictions (e.g., simulations) using the model and save results.
        """
        logger.info(
            f"--- Starting Prediction for Model : {self.model_id} on dataset : {self.dataset_id} ---"
        )

        trainer = self.trainer
        self.datamodule.setup(stage="predict")
        predict_dataloader = (
            self.datamodule.test_dataloader()
        )  # ou un dataloader spécifique

        trainer.predict(
            model=self.model,
            dataloaders=predict_dataloader,
            ckpt_path=self.checkpoint_path,
        )

        # Ensure the directory exists
        output_dir = os.path.dirname(self.dirpath)
        data_save_dir = os.path.join(output_dir, "distributions_comparisons")
        self.model.format_and_save_simulations(save_dir=data_save_dir)

        comparator = NTPPComparatorFactory.create_comparator(
            label_data=self.datamodule.test_dataset,
            simulation=self.model.simulations,
            num_event_types=self.datamodule.num_event_types,
            output_dir=data_save_dir,
        )

        logger.info(
            "Running comprehensive evaluation of temporal point process distributions..."
        )
        comparator.run_comprehensive_evaluation()
        logger.info(f"Simulations saved to {data_save_dir}")

        logger.info("Generating intensity graph...")
        self.model.intensity_graph(save_dir=data_save_dir)
