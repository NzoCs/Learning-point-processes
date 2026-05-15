import os
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Type, Union, Protocol, runtime_checkable

from pytorch_lightning.loggers import (
    CSVLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)
from pydantic import Field

from new_ltpp.configs.base_config import Config, ConfigValidationError
from new_ltpp.utils import logger


class LoggerType(StrEnum):
    CSV = "csv"
    WandB = "wandb"
    MLFLOW = "mlflow"
    COMET = "comet"
    NEPTUNE = "neptune"
    TENSORBOARD = "tensorboard"


class LoggerAdapter(ABC):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return []

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        for param in cls.get_required_params():
            if param not in config:
                raise ValueError(f"Parameter '{param}' is required for {cls.__name__}")
        return config

    @classmethod
    @abstractmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        pass


@runtime_checkable
class ILoggerAdapter(Protocol):
    @classmethod
    def get_required_params(cls) -> List[str]: ...
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]: ...
    @classmethod
    def configure(cls, config: Dict[str, Any]) -> Any: ...


class CSVLoggerAdapter(LoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["save_dir", "name"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> CSVLogger:
        os.makedirs(config["save_dir"], exist_ok=True)
        return CSVLogger(**config)


class WandBLoggerAdapter(LoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["project"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> WandbLogger:
        save_dir = config["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        import wandb

        wandb.finish()
        wandb.init(dir=save_dir)
        return WandbLogger(**config)


class MLflowLoggerAdapter(LoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["experiment_name"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> MLFlowLogger:
        save_dir = config.get("save_dir") or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)

        if "tracking_uri" not in config or not config.get("tracking_uri"):
            mlruns_path = Path(save_dir).resolve() / "mlruns"
            mlruns_path.mkdir(parents=True, exist_ok=True)
            config["tracking_uri"] = f"file://{mlruns_path.as_posix()}"

        allowed_keys = {
            "experiment_name",
            "tracking_uri",
            "run_name",
            "save_dir",
            "tags",
            "log_model",
            "nested",
            "run_id",
            "prefix",
            "artifact_location",
        }
        filtered_config = {k: v for k, v in config.items() if k in allowed_keys}
        try:
            return MLFlowLogger(**filtered_config)
        except Exception as e:
            logger.exception(
                "Failed to configure MLFlowLogger with config: %s", filtered_config
            )
            raise RuntimeError(
                f"Could not create MLFlowLogger. Original error: {e}. Check your tracking_uri and mlflow installation."
            )


class TensorboardLoggerAdapter(LoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["save_dir"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> TensorBoardLogger:
        if not config.get("name"):
            config["name"] = "tb_logs"
        return TensorBoardLogger(**config)


LOGGER_ADAPTERS: Dict[LoggerType, Type[LoggerAdapter]] = {
    LoggerType.CSV: CSVLoggerAdapter,
    LoggerType.WandB: WandBLoggerAdapter,
    LoggerType.MLFLOW: MLflowLoggerAdapter,
    LoggerType.TENSORBOARD: TensorboardLoggerAdapter,
}


class LoggerConfig(Config):
    """Configuration for logging in experiments."""

    save_dir: Union[str, Path]
    type: LoggerType = LoggerType.TENSORBOARD
    config: Dict[str, Any] = Field(default_factory=dict)

    def configure_logger(self) -> Any:
        adapter = LOGGER_ADAPTERS.get(self.type)
        if not adapter:
            raise ConfigValidationError(
                f"No adapter available for logger type: {self.type}"
            )
        self.config["save_dir"] = self.save_dir
        return adapter.configure(self.config.copy())


class LoggerFactory:
    @staticmethod
    def create_logger(config: Union[Dict[str, Any], LoggerConfig]) -> Any:
        logger_config = LoggerConfig(**config) if isinstance(config, dict) else config
        return logger_config.configure_logger()
