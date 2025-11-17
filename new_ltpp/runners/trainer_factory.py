import logging
import re
from pathlib import Path
from typing import List, Optional, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger as LightningLogger
from pytorch_lightning.strategies import DDPStrategy

from new_ltpp.configs.runner_config import TrainingConfig
from new_ltpp.utils import logger as console_logger


class CheckpointManager:
    """Manage best checkpoints and versioning."""

    BEST_PATTERN = re.compile(r"best(?:-v(\d+))?\.ckpt$")

    def __init__(self, dirpath: str):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def list(self) -> List[str]:
        return sorted(f.name for f in self.dirpath.glob("*.ckpt"))

    def _extract_version(self, filename: str) -> int:
        match = self.BEST_PATTERN.search(filename)
        if not match:
            return -1
        return int(match.group(1) or 0)

    def latest_best(self) -> Optional[str]:
        checkpoints = list(self.dirpath.glob("best*.ckpt"))
        if not checkpoints:
            console_logger.info(f"No 'best' checkpoints found in {self.dirpath}")
            return None

        checkpoints.sort(
            key=lambda ckpt: self._extract_version(ckpt.name),
            reverse=True,
        )

        latest = checkpoints[0]
        console_logger.info(f"Using latest best checkpoint: {latest.name}")
        return str(latest)

    def get(self, name: str) -> Optional[str]:
        path = self.dirpath / (name if name.endswith(".ckpt") else f"{name}.ckpt")
        if path.exists():
            console_logger.info(f"Using checkpoint: {path.name}")
            return str(path)
        console_logger.warning(f"Checkpoint not found: {path}")
        return None

import pytorch_lightning as pl


class PredictionStatsCallback(pl.Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called once before prediction begins."""
        if hasattr(pl_module, "init_statistics_collector"):
            pl_module.init_statistics_collector(output_dir=self.output_dir)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called once after prediction ends."""
        if hasattr(pl_module, "finalize_statistics"):
            pl_module.finalize_statistics()

        if hasattr(pl_module, "intensity_graph"):
            pl_module.intensity_graph(save_dir=self.output_dir)


class TrainerFactory:
    """
    Responsible for building a full PyTorch Lightning Trainer.
    Includes logger, callbacks, DDP, precision, matmul Tensor Core optimization.
    """

    # ============================================================
    #  MATMUL PRECISION SETUP (Tensor Cores)
    # ============================================================

    @staticmethod
    def _configure_matmul_precision(logger: Optional[logging.Logger] = console_logger):
        """
        Enable Tensor Core optimization when possible.
        """
        if not torch.cuda.is_available():
            return

        try:
            major, _ = torch.cuda.get_device_capability(0)
        except Exception:
            # In case the capability cannot be read
            return

        if major >= 7:
            torch.set_float32_matmul_precision("medium")
            if logger:
                logger.info(
                    f"Using Tensor Core optimization: torch.set_float32_matmul_precision('medium') "
                    f"(GPU capability {major}.x)"
                )
        else:
            if logger:
                logger.info(
                    f"GPU capability {major}.x < 7.0 → keeping default matmul precision"
                )

    # ============================================================
    #  DEVICE & PRECISION RESOLUTION
    # ============================================================

    @staticmethod
    def _resolve_devices(devices: Optional[int]):
        if isinstance(devices, int) and devices > 0:
            if torch.cuda.is_available():
                return devices, "gpu"
            else:
                return "auto", "auto"
        return "auto", "auto"

    @staticmethod
    def _resolve_precision(use_precision_16: bool):
        return "16-mixed" if use_precision_16 else "32-true"

    @staticmethod
    def _resolve_strategy(devices: Optional[int]):
        if isinstance(devices, int) and devices > 1:
            return DDPStrategy(find_unused_parameters=True)
        return "auto"

    # ============================================================
    #  CALLBACKS
    # ============================================================

    @staticmethod
    def _build_callbacks(
        training_config: TrainingConfig, checkpoints_dir: Path | str
    ):
        ckpt_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoints_dir,
            filename="best",
            save_top_k=1,
            mode="min",
            every_n_epochs=training_config.checkpoints_freq,
            auto_insert_metric_name=False,
            save_last=True,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=training_config.patience,
            verbose=True,
        )

        return [ckpt_callback, early_stop_callback]

    # ============================================================
    #  FACTORY ENTRYPOINT
    # ============================================================

    @classmethod
    def create(
        cls,
        training_config: TrainingConfig,
        checkpoints_dir: Path | str,
        trainer_logger: LightningLogger | None = None,
    ) -> pl.Trainer:
        """
        Create a full trainer from RunnerConfig.
        """

        # 2. matmul precision (Tensor Core optimization)
        cls._configure_matmul_precision()

        # 3. callbacks
        callbacks = cls._build_callbacks(
            training_config,
            checkpoints_dir
        )

        # 4. device resolution
        devices, accelerator = cls._resolve_devices(training_config.devices)

        # 5. DDP strategy
        strategy = cls._resolve_strategy(training_config.devices)

        # 6. precision
        precision = cls._resolve_precision(training_config.use_precision_16)

        # 7. final trainer
        trainer = pl.Trainer(
            max_epochs=training_config.max_epochs,
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            logger=trainer_logger,
            log_every_n_steps=training_config.log_freq,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=True,
            check_val_every_n_epoch=training_config.val_freq,
            precision=precision,
            accumulate_grad_batches=training_config.accumulate_grad_batches,
        )

        return trainer
