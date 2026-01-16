from pathlib import Path
from typing import Any, Dict, Union, cast, Self, TypedDict, List

from new_ltpp.configs.config_factory import ConfigType
from new_ltpp.configs.runner_config import RunnerConfig
from new_ltpp.utils import logger

from .base_config_builder import ConfigBuilder
from .data_config_builder import DataConfigBuilder
from .model_config_builder import ModelConfigBuilder
from .training_config_builder import TrainingConfigBuilder


class RunnerConfigDict(TypedDict):
    model_id: str | None
    model_config: Dict[str, Any] | None
    data_config: Dict[str, Any] | None
    training_config: Dict[str, Any] | None
    save_dir: Union[str, Path] | None
    logger_config: Dict[str, Any] | None
    enable_logging: bool | None


class RunnerConfigBuilder(ConfigBuilder):
    """
    Builder for RunnerConfig - orchestrates complete model training pipeline configuration.

    This builder delegates configuration to three sub-builders:
    - `data_builder` (DataConfigBuilder)
    - `model_builder` (ModelConfigBuilder)
    - `training_builder` (TrainingConfigBuilder)

    Design note:
    - The **RunnerConfigBuilder** exposes only a small surface for global
        fields (e.g. **save_dir**, **logger_config**). All dataset, model and
        training parameters should be set on the corresponding sub-builder. This
        keeps responsibilities clear and enables fluent chaining, for example
        `runner.data_builder.set_src_dir(...).set_batch_size(...)`.

    **Correct Usage (fluent chaining)**

    - Use the internal builders to configure their domain and then call
        `build(model_id=...)` on the runner builder.

    .. code-block:: python

            # Configure data, model and training via their builders (fluent)
            builder = RunnerConfigBuilder()
            (builder.data_builder
                            .set_dataset_id("my_dataset")
                            .set_src_dir("/data")
                            .set_batch_size(32))

            (builder.model_builder
                            .set_general_specs({"hidden_size": 64})
                            .set_model_specs({}))

            (builder.training_builder
                            .set_max_epochs(100)
                            .set_batch_size(32)
                            .set_lr(1e-3))

            config = builder.build(model_id="NHP")
    """

    _config_dict: RunnerConfigDict
    model_builder: ModelConfigBuilder
    data_builder: DataConfigBuilder
    training_builder: TrainingConfigBuilder

    def __init__(self):
        self._config_dict = {
            "model_id": None,
            "model_config": None,
            "data_config": None,
            "training_config": None,
            "save_dir": None,
            "logger_config": None,
            "enable_logging": None,
        }
        self.model_builder = ModelConfigBuilder()
        self.data_builder = DataConfigBuilder()
        self.training_builder = TrainingConfigBuilder()

    @property
    def config_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self._config_dict)

    @property
    def config_type(self) -> ConfigType:
        return ConfigType.RUNNER
    
    @property
    def required_fields(self) -> List[str]:
        return [
            "model_config",
            "data_config",
            "training_config",
            "model_id",
        ]

    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load config from a dictionary matching RunnerConfigDict structure."""
        self._config_dict = cast(RunnerConfigDict, config_dict)

        # Sync sub-builders
        if self._config_dict.get("training_config"):
            self.training_builder.from_dict(self.config_dict["training_config"])
        if self._config_dict.get("model_config"):
            self.model_builder.from_dict(self.config_dict["model_config"])
        if self._config_dict.get("data_config"):
            self.data_builder.from_dict(self.config_dict["data_config"])
    
    def build(self, **kwargs) -> RunnerConfig:
        """Build and return a RunnerConfig instance."""
        # Sync sub-builder configs
        self._config_dict["model_config"] = self.model_builder.config_dict
        self._config_dict["data_config"] = self.data_builder.config_dict
        self._config_dict["training_config"] = self.training_builder.config_dict

        clean_config_dict = {}

        if len(self.model_builder.get_unset_required_fields()) > 0:
            raise ValueError(
                f"Cannot build RunnerConfig, ModelConfig required fields not set: {self.model_builder.get_unset_required_fields()}"
            )
        clean_config_dict["model_config"] = self.model_builder.get_clean_dict()

        if len(self.data_builder.get_unset_required_fields()) > 0:
            raise ValueError(
                f"Cannot build RunnerConfig, DataConfig required fields not set: {self.data_builder.get_unset_required_fields()}"
            )
        clean_config_dict["data_config"] = self.data_builder.get_clean_dict()

        if len(self.training_builder.get_unset_required_fields()) > 0:
            raise ValueError(
                f"Cannot build RunnerConfig, TrainingConfig required fields not set: {self.training_builder.get_unset_required_fields()}"
            )
        clean_config_dict["training_config"] = self.training_builder.get_clean_dict()
        
        logger.info("Building RunnerConfig with:", self.config_dict)

        if len(self.get_unset_fields()) > 0:
            logger.info("Unset fields will be assigned default values:", self.get_unset_fields())

        # Include optional runner-level fields if provided
        if self.config_dict.get("save_dir") is not None:
            clean_config_dict["save_dir"] = self.config_dict.get("save_dir")
        if self.config_dict.get("logger_config") is not None:
            clean_config_dict["logger_config"] = self.config_dict.get("logger_config")
        if self.config_dict.get("enable_logging") is not None:
            clean_config_dict["enable_logging"] = self.config_dict.get("enable_logging")

        return RunnerConfig(**clean_config_dict, **kwargs)

    # RunnerConfig explicit parameters
    def set_save_dir(self, save_dir: Union[str, Path]) -> Self:
        """Set save directory for results.

        Args:
            save_dir: Path to save directory
        """
        self._config_dict["save_dir"] = save_dir
        return self

    # TrainingConfig convenience methods
    def set_max_epochs(self, max_epochs: int) -> Self:
        """Set maximum number of training epochs in training_config.

        Args:
            max_epochs: Maximum number of epochs
        """
        self.training_builder.set_max_epochs(max_epochs)
        return self
