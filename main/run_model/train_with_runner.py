import argparse

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run experiments with the new Runner class")
    
    parser.add_argument('--config_dir', type=str, required=False, default='./train_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')
    
    parser.add_argument('--experiment_id', type=str, required=False, default='NHP',
                        help='Experiment id in the config file.')
    
    parser.add_argument('--dataset_id', type=str, required=False, default='test',
                        help='Dataset id in the config file.')
    
    parser.add_argument('--phase', type=str, required=False, default='all',
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
    
    # Method 1: Using the Runner class directly (recommended for more control)
    print(f"=== Running experiment with phase: {args.phase} ===")
    runner = Runner(
        config=config,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir
    )

    runner.run(phase=args.phase)
    
    # You can also run specific phases individually:
    # runner = Runner(config)
    # runner.train()          # Only training (with logging)
    # runner.test()           # Only testing (without logging)
    # runner.predict()        # Only prediction (without logging)
    
    # Or run multiple specific phases:
    # results = runner.run(["train", "test"])  # Only train and test

if __name__ == '__main__':
    main()
