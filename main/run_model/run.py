import argparse

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def main():
    parser = argparse.ArgumentParser(description="Run experiments with the new Runner class")
    
    parser.add_argument('--config_dir', type=str, required=False, default='./runner_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')
    
    parser.add_argument('--experiment_id', type=str, required=False, default='THP',
                        help='Experiment id in the config file.')
    
    parser.add_argument('--dataset_id', type=str, required=False, default='H2expc',
                        help='Dataset id in the config file.')
    
    parser.add_argument('--phase', type=str, required=False, default='test',
                        choices=['train', 'test', 'predict', 'validation', 'all'],
                        help='Phase to execute: train, test, predict, validation, or all')
    
    parser.add_argument('--checkpoint_path', type=str, required=False, default=None,
                        help='Path to checkpoint file (without .ckpt extension)')
    
    parser.add_argument('--output_dir', type=str, required=False, default=None,
                        help='Output directory for saving results')

    args = parser.parse_args()

    # Build configuration from YAML using the utility
    config_dict = parse_runner_yaml_config(args.config_dir, args.experiment_id, args.dataset_id)
    
    config = RunnerConfig.from_dict(config_dict)

    runner = Runner(
        config=config,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir
    )
    runner.run(phase=args.phase)
    

if __name__ == '__main__':
    main()
