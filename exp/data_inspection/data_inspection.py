import argparse

from easy_tpp.config_factory import Config
from easy_tpp.preprocess.data_loader import TPPDataModule
from easy_tpp.preprocess.visualizer import Visualizer

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='./config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='hawkes1',
                        help='Experiment id in the config file.')

    args = parser.parse_args()
    
    config = Config.build_from_yaml_file(args.config_dir, experiment_id = args.experiment_id)
    
    data_module = TPPDataModule(config)
    visu = Visualizer(data_module = data_module, split = 'test')
    visu.show_all_distributions(log_scale = True)

if __name__ == '__main__':
    main()