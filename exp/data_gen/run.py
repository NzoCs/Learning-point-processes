import argparse

from easy_tpp.config_factory import Config
from easy_tpp.data_gen import BaseSimulator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='./config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='hawkes1',
                        help='Experiment id in the config file.')

    args = parser.parse_args()
    
    generator = BaseSimulator.generate_model_from_config(config)

    generator.generate_split()
    
if __name__ == '__main__' : 
    main()