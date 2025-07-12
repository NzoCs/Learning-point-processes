#!/usr/bin/env python3
"""
Simple benchmark example

Usage:
    python simple_benchmark.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def main() -> None:
    # List of models to benchmark
    models = ["NHP", "THP", "RMTPP"]
    dataset = "test"
    
    results = {}
    
    for model in models:
        print(f"Benchmarking {model}...")
        
        # Load configuration
        config_path = project_root / "main" / "run_model" / "runner_config.yaml"
        config_dict = parse_runner_yaml_config(str(config_path), model, dataset)
        config = RunnerConfig.from_dict(config_dict)
        
        # Run benchmark
        runner = Runner(config=config, output_dir=f"./benchmark_results/{model}")
        runner.run(phase="test")
        
        results[model] = "completed"
    
    print("Benchmark completed for all models:", list(results.keys()))


if __name__ == "__main__":
    main()
