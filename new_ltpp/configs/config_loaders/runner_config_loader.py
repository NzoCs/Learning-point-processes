from typing import Any, Dict, Optional, Union
from pathlib import Path

from .base_config_loader import ConfigLoader
from .training_config_loader import TrainingConfigYamlLoader
from .model_config_loader import ModelConfigYamlLoader
from .data_config_loader import DataConfigYamlLoader


class RunnerConfigYamlLoader(ConfigLoader):
    """Loader to extract RunnerConfig dictionary from YAML and sub-loaders."""

    def load(
        self,
        yaml_path: Union[str, Path],
        *,
        training_config_path: str,
        data_config_path: str,
        model_config_path: str,
        data_loading_config_path: str,
        data_specs_path: Optional[str] = None,
        simulation_config_path: Optional[str] = None,
        thinning_config_path: Optional[str] = None,
        logger_config_path: Optional[str] = None,
        general_specs_config_path: Optional[str] = None,
        model_specs_config_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Load raw data once
        config_data = self.load_yaml(yaml_path)

        training_loader = TrainingConfigYamlLoader()
        training_cfg = training_loader.load(
            yaml_path, training_config_path=training_config_path
        )

        model_loader = ModelConfigYamlLoader()
        model_cfg = model_loader.load(
            yaml_path,
            model_config_path=model_config_path,
            simulation_config_path=simulation_config_path,
            thinning_config_path=thinning_config_path,
            scheduler_config_path=None,  # handled below logic
            general_specs_path=general_specs_config_path,
            model_specs_path=model_specs_config_path,
        )

        # Special logic for scheduler config syncing
        if "max_epochs" not in model_cfg.get("scheduler_config", {}):
            scheduler_cfg = model_cfg.get("scheduler_config", {})
            if "max_epochs" not in scheduler_cfg:
                scheduler_cfg.update(
                    {
                        "lr_scheduler": training_cfg.get("lr_scheduler"),
                        "lr": training_cfg.get("lr"),
                        "max_epochs": training_cfg.get("max_epochs"),
                    }
                )
            model_cfg["scheduler_config"] = scheduler_cfg

        data_loader = DataConfigYamlLoader()
        data_cfg = data_loader.load(
            yaml_path,
            data_config_path=data_config_path,
            data_loading_config_path=data_loading_config_path,
            tokenizer_specs_path=data_specs_path,
        )

        runner_cfg = {
            "training_config": training_cfg,
            "model_config": model_cfg,
            "data_config": data_cfg,
            "save_dir": None,
            "logger_config": None,
        }

        if logger_config_path:
            # We already loaded file, so reuse logic
            runner_cfg["logger_config"] = self._get_nested_value(
                config_data, logger_config_path
            )

        return runner_cfg
