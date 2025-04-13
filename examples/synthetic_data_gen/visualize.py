import argparse

from easy_tpp.config_factory import Config
from easy_tpp.synthetic_data_generator import BaseGenerator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='./config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='H1exp',
                        help='Experiment id in the config file.')

    args = parser.parse_args()
    
    config = Config.build_from_yaml_file(yaml_dir = args.config_dir, experiment_id = args.experiment_id)
    
    generator = BaseGenerator.generate_model_from_config(config)

    generator.intensity_graph(plot=True, precision = 200)
    
if __name__ == '__main__' : 
    main()