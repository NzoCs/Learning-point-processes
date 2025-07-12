#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_dir', type=str, required=False, 
                       default='runner_config.yaml',
                       help='Configuration yaml file.')
    
    parser.add_argument('--experiment_id', type=str, required=False, 
                       default='NHP',
                       help='Experiment id in the config file.')
    
    parser.add_argument('--dataset_id', type=str, required=False, 
                       default='test',
                       help='Dataset id in the config file.')
    
    args = parser.parse_args()
    
    # Load configuration using modern API
    config_dict = parse_runner_yaml_config(args.config_dir, args.experiment_id, args.dataset_id)
    config = RunnerConfig.from_dict(config_dict)
    
    # Create and run - Pipeline complet avec prédictions
    runner = Runner(config=config, output_dir="./results/train_nhp")
    runner.run(phase="train")
    runner.run(phase="test") 
    runner.run(phase="predict")  # Génère simulations et comparaisons


if __name__ == '__main__':
    main()
