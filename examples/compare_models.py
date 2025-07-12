#!/usr/bin/env python3
"""
Simple model comparison example

Usage:
    python compare_models.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def run_experiment(model_name, dataset_name):
    """Run single experiment"""
    config_path = project_root / "main" / "run_model" / "runner_config.yaml"
    config_dict = parse_runner_yaml_config(str(config_path), model_name, dataset_name)
    config = RunnerConfig.from_dict(config_dict)
    
    runner = Runner(config=config, output_dir=f"./results/{model_name}_{dataset_name}")
    runner.run(phase="test")


def main():
    # Compare different models on same dataset
    models = ["NHP", "THP", "RMTPP"]
    dataset = "test"
    
    for model in models:
        print(f"Testing {model}...")
        run_experiment(model, dataset)


if __name__ == "__main__":
    main()
