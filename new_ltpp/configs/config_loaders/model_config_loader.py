from typing import Any, Dict, Optional, Union
from pathlib import Path

from .base_config_loader import BaseConfigLoader
from new_ltpp.utils import logger


class ModelConfigYamlLoader(BaseConfigLoader):
    """Loader to extract ModelConfig dictionary from YAML."""

    def load(
        self,
        yaml_path: Union[str, Path],
        *,
        model_config_path: Optional[str] = None,
        simulation_config_path: Optional[str] = None,
        thinning_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        general_specs_path: Optional[str] = None,
        model_specs_path: Optional[str] = None,
    ) -> Dict[str, Any]:  # type: ignore[override]
        """
        Load model configuration from a YAML file.

        Args:
            yaml_path: Path to YAML file.
            model_config_path: Path to model config (backward compatibility).
            simulation_config_path: Path to simulation config.
            thinning_config_path: Path to thinning config.
            scheduler_config_path: Path to scheduler config.
            general_specs_path: Path to general specs.
            model_specs_path: Path to model specs.

        Returns:
            A dictionary matching ModelConfigDict structure.
        """
        data = self.load_yaml(yaml_path)

        # Load model_cfg only if model_config_path is provided (backward compatibility)
        model_cfg = {}
        if model_config_path:
            model_cfg = self._get_nested_value(data, model_config_path)

        simulation_cfg = (
            self._get_nested_value(data, simulation_config_path)
            if simulation_config_path
            else {}
        )
        scheduler_cfg = (
            self._get_nested_value(data, scheduler_config_path)
            if scheduler_config_path
            else {}
        )
        thinning_cfg = (
            self._get_nested_value(data, thinning_config_path)
            if thinning_config_path
            else {}
        )

        # Load general_specs and model_specs from specific paths or from model_cfg
        if general_specs_path:
            general_specs = self._get_nested_value(data, general_specs_path)
        else:
            general_specs = model_cfg.get("general_specs", {})

        if model_specs_path:
            model_specs = self._get_nested_value(data, model_specs_path)
        else:
            model_specs = model_cfg.get("model_specs", {})

        config_dict = {}
        config_dict["general_specs"] = general_specs
        config_dict["model_specs"] = model_specs
        config_dict["simulation_config"] = simulation_cfg
        config_dict["thinning_config"] = thinning_cfg
        config_dict["scheduler_config"] = scheduler_cfg

        # Merge other fields from model_cfg if present
        for key in [
            "device",
            "gpu",
            "is_training",
            "compute_simulation",
            "pretrain_model_path",
        ]:
            if key in model_cfg:
                config_dict[key] = model_cfg[key]

        # Smart loading for num_mc_samples (required field)
        if "num_mc_samples" in model_cfg:
            config_dict["num_mc_samples"] = model_cfg["num_mc_samples"]
        elif "num_mc_samples" in general_specs:
            config_dict["num_mc_samples"] = general_specs["num_mc_samples"]
        else:
            # Fallback default if not specified anywhere
            logger.info("num_mc_samples not found in config, defaulting to 100")
            config_dict["num_mc_samples"] = 100

        return config_dict
