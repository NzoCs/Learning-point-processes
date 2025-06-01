import argparse

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def main():
    parser = argparse.ArgumentParser(description="Run experiments with the new Runner class")
    
    parser.add_argument('--config_dir', type=str, required=False, default='./runner_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')
    
    parser.add_argument('--experiment_id', type=str, required=False, default='IntensityFree',
                        help='Experiment id in the config file.')
    
    parser.add_argument('--dataset_id', type=str, required=False, default='test',
                        help='Dataset id in the config file.')
    
    parser.add_argument('--phase', type=str, required=False, default='predict',
                        choices=['train', 'test', 'predict', 'validation', 'all'],
                        help='Phase to execute: train, test, predict, validation, or all')
    
    parser.add_argument('--checkpoint_path', type=str, required=False, default=None,
                        help='Path to checkpoint file (without .ckpt extension)')
    
    parser.add_argument('--output_dir', type=str, required=False, default=None,
                        help='Output directory for saving results')

    args = parser.parse_args()
    
    # Build configuration
    config = Config.build_from_yaml_file(
        yaml_dir=args.config_dir, 
        experiment_id=args.experiment_id, 
        dataset_id=args.dataset_id
    )
    
    runner = Runner(
        config=config,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir
    )
    
    runner.run(phase=args.phase)
    

if __name__ == '__main__':
    main()
