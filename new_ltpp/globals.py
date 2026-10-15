from pathlib import Path

import yaml_configs

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_FILE = Path(yaml_configs.__file__).resolve().parent / "configs.yaml"
# Outputs belong to the user's working directory, even for a wheel installation.
OUTPUT_DIR = Path.cwd() / "artifacts"
