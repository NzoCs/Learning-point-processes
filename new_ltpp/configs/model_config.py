from typing import Optional, Union
from pydantic import ConfigDict, Field, PositiveInt, PositiveFloat, model_validator
from pathlib import Path

from new_ltpp.utils import logger
from .base_config import Config
from .config_utils import load_yaml, extract


def get_available_gpu() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        return -1
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU")
        return -1


class SchedulerConfig(Config):
    max_epochs: PositiveInt
    lr: PositiveFloat = 1e-3
    lr_scheduler: bool = True


class ThinningConfig(Config):
    num_sample: PositiveInt
    num_exp: PositiveInt
    num_samples_boundary: PositiveInt = 200
    over_sample_rate: float = Field(default=2.0, gt=1.0)

    @model_validator(mode="before")
    @classmethod
    def setup_boundary(cls, values: dict) -> dict:
        if values.get("num_samples_boundary") is None:
            values["num_samples_boundary"] = values.get("num_sample")
        return values


class ModelSpecsConfig(Config):
    hidden_size: Optional[int] = None
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="allow")


class ModelConfig(Config):
    model_id: str
    is_training: bool = True
    compute_simulation: bool = False
    compute_metrics: bool = True
    device: str = "cpu"
    gpu: int = Field(default_factory=get_available_gpu)

    specs: ModelSpecsConfig

    scheduler_config: SchedulerConfig
    thinning_config: ThinningConfig

    @classmethod
    def from_yaml_components(
        cls,
        yaml_path: Union[str, Path],
        model_id: str,
        model_specs_path: Optional[str] = None,
        general_specs_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        thinning_config_path: Optional[str] = None,
    ) -> "ModelConfig":
        data = load_yaml(yaml_path)

        specs_dict = {}
        for path in filter(None, [model_specs_path, general_specs_path]):
            try:
                specs_dict.update(extract(data, path))
            except KeyError:
                pass

        extra: dict = {}
        if scheduler_config_path:
            try:
                extra["scheduler_config"] = SchedulerConfig(
                    **extract(data, scheduler_config_path)
                )
            except KeyError:
                pass
        if thinning_config_path:
            try:
                extra["thinning_config"] = ThinningConfig(
                    **extract(data, thinning_config_path)
                )
            except KeyError:
                pass

        return cls(model_id=model_id, specs=ModelSpecsConfig(**specs_dict), **extra)
