from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from new_ltpp.configs.base_config import Config
from new_ltpp.configs.runner_config import RunnerConfig
from new_ltpp.utils import parse_uri_to_protocol_and_path


@dataclass
class HPOConfig(Config):
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

    @property
    def storage_protocol(self):
        storage_protocol, _ = parse_uri_to_protocol_and_path(self.storage_uri)
        return storage_protocol

    @property
    def storage_path(self):
        _, storage_path = parse_uri_to_protocol_and_path(self.storage_uri)
        return storage_path


@dataclass
class HPORunnerConfig(Config):
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
