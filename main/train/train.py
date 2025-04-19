import argparse

from easy_tpp.config_factory import Config
from easy_tpp.runner import Trainer



def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='./train_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='NHP_train',
                        help='Experiment id in the config file.')
    
    parser.add_argument('--dataset_id', type=str, required=False, default='test',
                        help='Dataset id in the config file.')

    args = parser.parse_args()
    
    config = Config.build_from_yaml_file(yaml_dir = args.config_dir, experiment_id = args.experiment_id, dataset_id = args.dataset_id)
    
    plrunner = Trainer(config)
    
    plrunner.train()
    
    
if __name__ == '__main__' :
    main()