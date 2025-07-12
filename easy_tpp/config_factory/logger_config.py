from enum import Enum
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
import os
from contextlib import contextmanager
from pytorch_lightning.loggers import (
    WandbLogger,
    TensorBoardLogger,
    CSVLogger,
    MLFlowLogger,
    NeptuneLogger,
    CometLogger,  # for example, for Comet.ml
)
from easy_tpp.utils import logger
from dataclasses import dataclass, field
from easy_tpp.config_factory.base import (
    BaseConfig,
    ConfigValidationError,
    config_factory,
    config_class,
)


class LoggerType(Enum):
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
        try:
            import wandb

            wandb.finish()
            wandb.init(dir=save_dir)
            return WandbLogger(**config)
        except Exception:
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
        return MLFlowLogger(**config)


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
LOGGER_ADAPTERS = {
    LoggerType.CSV: CSVLoggerAdapter,
    LoggerType.WandB: WandBLoggerAdapter,
    LoggerType.MLFLOW: MLflowLoggerAdapter,
    LoggerType.COMET: CometLoggerAdapter,
    LoggerType.NEPTUNE: NeptuneLoggerAdapter,
    LoggerType.TENSORBOARD: TensorboardLoggerAdapter,
}


@config_class("logger_config")
@dataclass
class LoggerConfig(BaseConfig):
    """
    Configuration for logging in experiments. This class allows you to specify the type of logger
    (e.g., CSV, WandB, MLFlow) and its parameters. It also provides methods to validate and configure
    the logger based on the specified type.
    Args:
        save_dir (Optional[str]): Directory where logs will be saved.
        logger_type (Optional[LoggerType]): Type of logger to use (e.g., CSV, WandB, MLFlow, Comet, Neptune, TensorBoard).
        config (Dict[str, Any]): Additional configuration parameters for the logger.
    """

    save_dir: Optional[str] = None
    logger_type: Optional[LoggerType] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert string logger_type to LoggerType if needed
        if isinstance(self.logger_type, str):
            try:
                self.logger_type = LoggerType(self.logger_type)
            except ValueError:
                raise ConfigValidationError(
                    f"Unknown logger type: {self.logger_type}", "logger_type"
                )
        # Get the adapter for this logger type
        self.adapter = LOGGER_ADAPTERS.get(self.logger_type)
        if not self.adapter:
            logger.warning(f"No adapter available for logger type: {self.logger_type}")
            logger.warning("The logger will not be configured.")
        # Validate and initialize the configuration
        if self.adapter is not None:
            self.config = self.adapter.validate_config(self.config)
            # If save_dir is provided, ensure it's in config
            if self.save_dir is not None:
                self.config["save_dir"] = self.save_dir
        super().__post_init__()

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "logger_type": self.logger_type.value if self.logger_type else None,
            "logger_config": self.config,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoggerConfig":
        # Accept both 'logger_type' and 'type' as keys
        logger_type = config_dict.get("logger_type") or config_dict.get("type")
        config = dict(
            config_dict.get("logger_config", {})
        )  # copy to avoid mutating input
        # Inject save_dir if present at the top level
        save_dir = config_dict.get("save_dir")
        if save_dir is not None:
            config["save_dir"] = save_dir
        # Filter out invalid keys for LoggerConfig
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        # Set logger_type and config explicitly
        filtered["logger_type"] = logger_type
        filtered["config"] = config
        dropped = set(config_dict.keys()) - valid_keys
        if dropped:
            logger.warning(f"Filtered out invalid LoggerConfig keys: {dropped}")
        return cls(**filtered)

    @staticmethod
    def parse_from_yaml_config(config, **kwargs):
        if kwargs:
            config.update(kwargs)
        return LoggerConfig.from_dict(config)

    def configure_logger(self) -> Any:
        config = self.config.copy()
        return self.adapter.configure(config)

    @classmethod
    def list_required_params(cls, logger_type: Union[LoggerType, str]) -> List[str]:
        if isinstance(logger_type, str):
            try:
                logger_type = LoggerType(logger_type)
            except ValueError:
                raise ValueError(f"Unknown logger type: {logger_type}")
        adapter = LOGGER_ADAPTERS.get(logger_type)
        if not adapter:
            raise ValueError(f"No adapter available for logger type: {logger_type}")
        return adapter.get_required_params()

    def get_required_fields(self):
        return []
