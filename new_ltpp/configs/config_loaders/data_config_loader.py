from typing import Any, Dict, Optional, Union
from pathlib import Path

from .base_config_loader import ConfigLoader


class DataConfigYamlLoader(ConfigLoader):
    """Loader to extract DataConfig dictionary from YAML."""

    def load(
        self,
        yaml_path: Union[str, Path],
        *,
        data_config_path: str,
        data_loading_config_path: Optional[str] = None,
        tokenizer_specs_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load data configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML file.
            data_config_path: Dotted path to the main data config section.
            data_loading_config_path: Optional path to data loading specs.
            tokenizer_specs_path: Optional path to tokenizer specs.

        Returns:
            A dictionary matching DataConfigDict structure.
        """
        config_dict = self.load_yaml(yaml_path)

        data_cfg = self._get_nested_value(config_dict, data_config_path)

        # Ensure dataset_id exists
        if isinstance(data_cfg, dict) and "dataset_id" not in data_cfg:
            dataset_id = data_config_path.split(".")[-1]
            data_cfg["dataset_id"] = dataset_id

        # Use DataLoadingSpecsBuilder if requested
        if data_loading_config_path:
            dl_cfg = self._get_nested_value(config_dict, data_loading_config_path)
            data_cfg.setdefault("data_loading_specs", dl_cfg)

        # Merge tokenizer_specs if requested
        if tokenizer_specs_path:
            specs_cfg = self._get_nested_value(config_dict, tokenizer_specs_path)
            data_cfg.setdefault("tokenizer_specs", specs_cfg)

        # Edge case: if src_dir is present and any required dir is missing, set all to src_dir
        required_dirs = ["train_dir", "valid_dir", "test_dir"]
        if "src_dir" in data_cfg:
            for d in required_dirs:
                if d not in data_cfg:
                    data_cfg[d] = data_cfg["src_dir"]
            data_cfg.pop("src_dir", None)

        return data_cfg
