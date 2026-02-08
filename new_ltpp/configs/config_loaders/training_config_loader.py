from typing import Any, Dict, Union
from pathlib import Path

from .base_config_loader import ConfigLoader


class TrainingConfigYamlLoader(ConfigLoader):
    """Loader to extract TrainingConfig dictionary from YAML."""

    def load(
        self,
        yaml_path: Union[str, Path],
        *,
        training_config_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load training configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML file.
            training_config_path: Dotted path to the training config section.

        Returns:
            A dictionary matching TrainingConfigDict structure.
        """
        config_dict = self.load_yaml(yaml_path)
        training_cfg = self._get_nested_value(config_dict, training_config_path)
        return training_cfg
