#!/usr/bin/env python3
"""
Complete example with predictions and distribution analysis

Usage:
    python prediction_analysis.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def train_and_analyze_model(model_name: str, dataset_name: str) -> None:
    """Trains a model and generates distribution analysis."""
    print(f"🧠 {model_name} on {dataset_name}")

    # Configuration
    config_path = Path(__file__).parent / "runner_config.yaml"
    config_dict = parse_runner_yaml_config(str(config_path), model_name, dataset_name)
    config = RunnerConfig.from_dict(config_dict)

    # Runner
    output_dir = f"./analysis_results/{model_name}_{dataset_name}"
    runner = Runner(config=config, output_dir=output_dir)

    # Complete pipeline
    runner.run(phase="train")
    runner.run(phase="test")
    runner.run(phase="predict")

    print(f"✅ {model_name} completed")


def compare_models_predictions() -> None:
    """Compare predictions from multiple models."""
    models = ["NHP", "THP", "RMTPP"]
    dataset = "test"

    print("🔬 Comparing models")

    for model in models:
        try:
            train_and_analyze_model(model, dataset)
        except Exception as e:
            print(f"❌ Error {model}: {str(e)[:30]}...")


def analyze_synthetic_vs_real() -> None:
    """Compare performance on synthetic vs real data."""
    model = "NHP"
    datasets = ["test", "synthetic_hawkes"]

    print("🎲 Synthetic vs real")

    for dataset in datasets:
        try:
            train_and_analyze_model(model, dataset)
        except Exception as e:
            print(f"❌ Error {dataset}: {str(e)[:30]}...")


def main() -> None:
    print("🚀 Prediction analysis")

    # 1. Model comparison
    compare_models_predictions()

    # 2. Synthetic vs real analysis
    analyze_synthetic_vs_real()

    print("🎉 Done!")


if __name__ == "__main__":
    main()
