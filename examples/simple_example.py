#!/usr/bin/env python3
"""
Complete EasyTPP pipeline example with prediction and analysis

Usage:
    python simple_example.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def main() -> None:
    # Load configuration
    config_path = Path(__file__).parent / "runner_config.yaml"
    config_dict = parse_runner_yaml_config(str(config_path), "NHP", "test")
    config = RunnerConfig.from_dict(config_dict)

    # Create runner
    runner = Runner(config=config, output_dir="./results/complete_pipeline")

    # Run complete pipeline: train -> test -> predict
    print("🚀 Lancement du pipeline complet...")

    # 1. Training
    print("📚 Phase d'entraînement...")
    runner.run(phase="train")

    # 2. Testing
    print("🧪 Phase de test...")
    runner.run(phase="test")

    # 3. Prediction and distribution comparison
    print("🔮 Phase de prédiction et comparaison des distributions...")
    runner.run(phase="predict")


if __name__ == "__main__":
    main()
