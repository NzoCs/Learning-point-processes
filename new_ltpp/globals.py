from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_FILE = ROOT_DIR / "yaml_configs" / "configs.yaml"
OUTPUT_DIR = ROOT_DIR / "artifacts"