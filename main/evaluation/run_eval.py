import argparse
import os

from easy_tpp.config_factory import Config
from easy_tpp.evaluate.evaluation_btw_data import Evaluation

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='./eval_config.yaml',
                        help='Dir of configuration yaml for dataset evaluation.')

    parser.add_argument('--experiment_id', type=str, required=False, default='H2expi_vs_H2expc',
                        help='Experiment id in the config file.')

    args = parser.parse_args()
    
    # Load the configuration
    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
    
    # Initialize the evaluator with the configuration
    evaluator = Evaluation(config)
    
    # Run the evaluation
    evaluator.run_evaluation()

if __name__ == '__main__':
    main()
