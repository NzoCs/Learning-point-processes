from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import Field, PositiveInt, model_validator, ConfigDict

from new_ltpp.configs.base_config import Config
from new_ltpp.configs.data_config import DataConfig
from new_ltpp.configs.logger_config import LoggerConfig, LoggerType
from new_ltpp.configs.model_config import ModelConfig
from new_ltpp.configs.statistical_test_config import (
    StatisticalTestConfig,
    SimulationConfig,
)
from new_ltpp.globals import OUTPUT_DIR


def detect_available_devices() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return -1


class TrainingConfig(Config):
    """Configuration for the Training."""

    max_epochs: PositiveInt
    lr: float = 1e-3
    lr_scheduler: bool = True
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    val_freq: PositiveInt = 10
    patience: PositiveInt = 20
    log_freq: PositiveInt = 5
    checkpoints_freq: PositiveInt = 5
    accumulate_grad_batches: PositiveInt = 1
    use_precision_16: bool = False
    devices: int = Field(default_factory=detect_available_devices)


class RunnerConfig(Config):
    """Configuration for the Runner."""

    model_id: str

    training_config: TrainingConfig
    model_cfg: ModelConfig = Field(alias="model_config")
    data_config: DataConfig
    simulation_config: SimulationConfig
    statistical_test_config: StatisticalTestConfig
    logger_config: Optional[LoggerConfig] = None

    enable_logging: bool = True

    # Computed fields
    save_dir: Union[str, Path] = OUTPUT_DIR
    dataset_id: str = ""
    base_dir: Path = Field(default_factory=Path)
    checkpoints_dir: Path = Field(default_factory=Path)
    model_dir: str = ""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def setup_directories(cls, values: dict) -> dict:
        data_config = values.get("data_config")
        model_cfg = values.get("model_config") or values.get("model_cfg")  # alias !

        if not data_config or not model_cfg:
            return values

        values["dataset_id"] = data_config.dataset_id

        specs_str = "_".join(
            f"{k}_{v}"
            for k, v in model_cfg.specs.model_dump(exclude={"model_config"}).items()
            if not k.startswith("_") and v is not None
        )

        root_dir = Path(values["save_dir"]) if values.get("save_dir") else OUTPUT_DIR
        base_dir = root_dir / values["dataset_id"] / f"{values['model_id']}_{specs_str}"
        checkpoints_dir = base_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        values["base_dir"] = base_dir
        values["checkpoints_dir"] = checkpoints_dir
        values["model_dir"] = str(checkpoints_dir)
        values["save_dir"] = str(base_dir / "logs")

        # Défaut dérivé du save_dir calculé
        logger_config = values.get("logger_config")
        if logger_config is None:
            values["logger_config"] = LoggerConfig(
                save_dir=values["save_dir"], type=LoggerType.TENSORBOARD
            )
        else:
            values["logger_config"] = logger_config.model_validate(
                {**logger_config.model_dump(), "save_dir": values["save_dir"]}
            )

        return values

    @classmethod
    def from_yaml_presets(
        cls,
        yaml_path: Union[str, Path],
        config_paths: Dict[str, str],
        model_id: str,
        **overrides,
    ) -> "RunnerConfig":
        data_cfg_path = config_paths["data_config_path"]
        dataset_id = data_cfg_path.split(".")[-1]

        data_config = DataConfig.from_yaml_components(
            yaml_path,
            dataset_id=dataset_id,
            data_config_path=data_cfg_path,
            data_loading_config_path=config_paths["data_loading_config_path"],
        )

        training_config = TrainingConfig.from_yaml(
            yaml_path, config_paths["training_config_path"]
        )

        if model_config_path := config_paths.get("model_config_path"):
            model_config = ModelConfig.from_yaml(yaml_path, model_config_path)
        else:
            model_config = ModelConfig.from_yaml_components(
                yaml_path,
                model_id=model_id,
                model_specs_path=config_paths.get("model_specs_config_path"),
                general_specs_path=config_paths.get("general_specs_config_path"),
                scheduler_config_path=config_paths.get("training_config_path"),
                thinning_config_path=config_paths.get("thinning_config_path"),
            )

        simulation_config = SimulationConfig.from_yaml(
            yaml_path, config_paths["simulation_config_path"]
        )

        statistical_test_config = StatisticalTestConfig.from_yaml_components(
            yaml_path,
            num_event_types=data_config.num_event_types,
            config_path=config_paths["statistical_test_config_path"],
        )

        # Apply max_epochs override
        if "max_epochs" in overrides:
            training_config = TrainingConfig(
                **{
                    **training_config.model_dump(),
                    "max_epochs": overrides.pop("max_epochs"),
                }
            )

        return cls(
            model_id=model_id,
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            simulation_config=simulation_config,
            statistical_test_config=statistical_test_config,
            **overrides,
        )

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "training_config": self.training_config.model_dump(mode="json"),
            "model_config": self.model_cfg.model_dump(mode="json"),
            "data_config": self.data_config.model_dump(mode="json"),
            "statistical_test_config": self.statistical_test_config.model_dump(
                mode="json"
            ),
        }
