from easy_tpp.config_factory.base import BaseConfig
from easy_tpp.config_factory.runner_config import RunnerConfig
from easy_tpp.utils import parse_uri_to_protocol_and_path, py_assert
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class HPOConfig(BaseConfig):
    framework_id: str = "optuna"
    storage_uri: Optional[str] = None
    is_continuous: bool = True
    num_trials: int = 50
    num_jobs: int = 1

    def get_required_fields(self) -> List[str]:
        return [
            "framework_id",
            "storage_uri",
            "is_continuous",
            "num_trials",
            "num_jobs",
        ]

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "framework_id": self.framework_id,
            "storage_uri": self.storage_uri,
            "is_continuous": self.is_continuous,
            "num_trials": self.num_trials,
            "num_jobs": self.num_jobs,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HPOConfig":
        return cls(
            framework_id=config_dict.get("framework_id", "optuna"),
            storage_uri=config_dict.get("storage_uri"),
            is_continuous=config_dict.get("is_continuous", True),
            num_trials=config_dict.get("num_trials", 50),
            num_jobs=config_dict.get("num_jobs", 1),
        )

    @property
    def storage_protocol(self):
        storage_protocol, _ = parse_uri_to_protocol_and_path(self.storage_uri)
        return storage_protocol

    @property
    def storage_path(self):
        _, storage_path = parse_uri_to_protocol_and_path(self.storage_uri)
        return storage_path

    def copy(self):
        return HPOConfig.from_dict(self.get_yaml_config())

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        if yaml_config is None:
            return None
        else:
            return HPOConfig.from_dict({**yaml_config, **kwargs})


@dataclass
class HPORunnerConfig(BaseConfig):
    hpo_config: HPOConfig = None
    runner_config: RunnerConfig = None

    def get_required_fields(self) -> List[str]:
        return ["hpo_config", "runner_config"]

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "hpo_config": (
                self.hpo_config.get_yaml_config() if self.hpo_config else None
            ),
            "runner_config": (
                self.runner_config.get_yaml_config() if self.runner_config else None
            ),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HPORunnerConfig":
        hpo_cfg = config_dict.get("hpo_config")
        if isinstance(hpo_cfg, dict):
            hpo_cfg = HPOConfig.from_dict(hpo_cfg)
        runner_cfg = config_dict.get("runner_config")
        if isinstance(runner_cfg, dict):
            runner_cfg = RunnerConfig.from_dict(runner_cfg)
        return cls(hpo_config=hpo_cfg, runner_config=runner_cfg)

    def copy(self):
        return HPORunnerConfig.from_dict(self.get_yaml_config())

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        runner_config = RunnerConfig.parse_from_yaml_config(yaml_config, **kwargs)
        hpo_config = HPOConfig.parse_from_yaml_config(yaml_config.get("hpo"), **kwargs)
        py_assert(
            hpo_config is not None,
            ValueError,
            "No hpo configs is provided for HyperTuner",
        )
        return HPORunnerConfig(hpo_config=hpo_config, runner_config=runner_config)
