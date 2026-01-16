import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Type, Union, cast

from pytorch_lightning.loggers import (
    CometLogger,
    CSVLogger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)

from new_ltpp.configs.base_config import Config, ConfigValidationError
from new_ltpp.utils import logger


class LoggerType(StrEnum):
    CSV = "csv"
    WandB = "wandb"
    MLFLOW = "mlflow"
    COMET = "comet"
    NEPTUNE = "neptune"
    TENSORBOARD = "tensorboard"


class BaseLoggerAdapter(ABC):
    """Abstract class defining the interface for all logger adapters, see the documentation of different loggers to
    understand the various parameters."""

    @classmethod
    def get_required_params(cls) -> List[str]:
        """Returns the list of required parameters for this logger"""
        return []

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and completes the configuration"""

        # Check required parameters
        for param in cls.get_required_params():
            if param not in config:
                raise ValueError(f"Parameter '{param}' is required for {cls.__name__}")

        return config

    @classmethod
    @abstractmethod
    def configure(cls, config: Dict[str, Any]) -> Any:
        """Configures and returns a logger instance"""
        pass


class CSVLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        """
        Returns the list of required parameters for CSVLogger.
        """
        return ["save_dir", "name"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> CSVLogger:
        """
        Configures and returns an instance of CSVLogger using the provided parameters.
        """
        # Create the save directory if necessary
        os.makedirs(config["save_dir"], exist_ok=True)

        return CSVLogger(**config)


class WandBLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        """
        Returns the list of required parameters for WandbLogger.
        """
        return ["project"]  # 'project' is the only strictly required parameter

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> WandbLogger:
        """
        Configures and returns an instance of WandbLogger using the provided parameters.
        """
        save_dir = config["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        import wandb

        wandb.finish()
        wandb.init(dir=save_dir)
        return WandbLogger(**config)


class MLflowLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        """
        Returns the list of required parameters for MLFlowLogger.
        """
        return ["experiment_name"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> MLFlowLogger:
        """
        Checks required parameters, configures and returns an instance of MLFlowLogger
        using the provided parameters.
        """
        # Ensure save_dir exists
        save_dir = config.get("save_dir") or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)

        # If no tracking URI is provided, default to a local file-based mlflow store
        if "tracking_uri" not in config or not config.get("tracking_uri"):
            # create a local mlruns directory inside the save_dir
            mlruns_path = Path(save_dir).resolve() / "mlruns"
            mlruns_path.mkdir(parents=True, exist_ok=True)
            config["tracking_uri"] = f"file://{mlruns_path.as_posix()}"

        # Allow only keys that MLFlowLogger typically accepts to avoid unexpected blocking
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
            # Provide a clearer error describing what was attempted
            logger.exception(
                "Failed to configure MLFlowLogger with config: %s", filtered_config
            )
            raise RuntimeError(
                f"Could not create MLFlowLogger. Original error: {e}. Check your tracking_uri and mlflow installation."
            )


class CometLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["api_key", "project_name"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> CometLogger:

        return CometLogger(**config)


class NeptuneLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["api_token", "project"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> NeptuneLogger:

        return NeptuneLogger(**config)


class TensorboardLoggerAdapter(BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["save_dir"]

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> TensorBoardLogger:

        if not config.get("name"):
            config["name"] = "tb_logs"

        return TensorBoardLogger(**config)


# Registry of adapters
LOGGER_ADAPTERS: Dict[LoggerType, Type[BaseLoggerAdapter]] = {
    LoggerType.CSV: CSVLoggerAdapter,
    LoggerType.WandB: WandBLoggerAdapter,
    LoggerType.MLFLOW: MLflowLoggerAdapter,
    LoggerType.COMET: CometLoggerAdapter,
    LoggerType.NEPTUNE: NeptuneLoggerAdapter,
    LoggerType.TENSORBOARD: TensorboardLoggerAdapter,
}


@dataclass
class LoggerConfig(Config):
    """
    Configuration for logging in experiments.

    Args:
        save_dir (str): Directory where logs will be saved.
        type (LoggerType): Type of logger to use. Defaults to TENSORBOARD.
        config (Dict[str, Any]): Additional configuration parameters for the logger.
    """

    save_dir: str | Path
    type: LoggerType = LoggerType.TENSORBOARD
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert string type to LoggerType if needed

        if isinstance(self.type, str):
            try:
                self.type = LoggerType(self.type)
            except ValueError:
                raise ConfigValidationError(f"Unknown logger type: {self.type}")

        # Get the adapter for this logger type
        adapter = LOGGER_ADAPTERS.get(self.type)
        if not adapter:
            raise ConfigValidationError(
                f"No adapter available for logger type: {self.type}",
                "logger_type",
            )

        # Prepare config with save_dir
        self.config = dict(self.config)
        self.config["save_dir"] = str(self.save_dir)

        # Validate the configuration with the adapter
        self.config = adapter.validate_config(self.config)

        super().__post_init__()

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "save_dir": str(self.save_dir),
            "logger_type": self.type.value,
            "config": self.config,
        }

    def configure_logger(self) -> Any:
        """
        Configure and return a logger instance.

        Returns:
            Logger instance
        """
        adapter = LOGGER_ADAPTERS.get(self.type)
        if not adapter:
            raise ConfigValidationError(
                f"No adapter available for logger type: {self.type}",
                "logger_type",
            )

        config = self.config.copy()
        return adapter.configure(config)

    @classmethod
    def list_required_params(cls, type: Union[LoggerType, str]) -> List[str]:
        if isinstance(type, str):
            try:
                type = LoggerType(type)
            except ValueError:
                raise ValueError(f"Unknown logger type: {type}")
        adapter = LOGGER_ADAPTERS.get(type)
        if not adapter:
            raise ValueError(f"No adapter available for logger type: {type}")
        return adapter.get_required_params()

    def get_required_fields(self):
        return ["save_dir"]


class LoggerFactory:
    @staticmethod
    def create_logger(config: Union[Dict[str, Any], LoggerConfig]) -> Any:
        """
        Créer une instance de logger à partir de la configuration.

        Args:
            config: Configuration du logger

        Returns:
            Instance du logger
        """
        logger_config = LoggerConfig(**cast(Dict[str, Any], config)) if isinstance(config, Dict[str, Any]) else cast(LoggerConfig, config)
        return logger_config.configure_logger()
