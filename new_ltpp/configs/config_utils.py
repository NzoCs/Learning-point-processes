from pathlib import Path
from typing import Union
import yaml


def load_yaml(yaml_path: Union[str, Path]) -> dict:
    with open(Path(yaml_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract(data: dict, path: str, sep: str = ".") -> dict:
    """Navigate nested dict via a separated path string.

    Example:
        extract(data, "data_configs.taxi")
        extract(data, "data_configs/taxi", sep="/")
    """
    for key in path.split(sep):
        if not isinstance(data, dict) or key not in data:
            raise KeyError(f"Key {key!r} not found at path {path!r}")
        data = data[key]
    return data


def extract_from_yaml(yaml_path: Union[str, Path], path: str, sep: str = ".") -> dict:
    """Load a YAML and extract a nested value in one call."""
    return extract(load_yaml(yaml_path), path, sep)
