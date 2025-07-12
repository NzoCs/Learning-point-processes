import argparse

from easy_tpp.config_factory import Config
from easy_tpp.data.preprocess.data_loader import TPPDataModule
from easy_tpp.data.preprocess.visualizer import Visualizer
import os


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_dir",
        type=str,
        required=False,
        default="./config.yaml",
        help="Dir of configuration yaml to train and evaluate the model.",
    )

    parser.add_argument(
        "--experiment_id",
        type=str,
        required=False,
        default="H2expi",
        help="Experiment id in the config file.",
    )

    args = parser.parse_args()

    config = Config.build_from_yaml_file(
        args.config_dir, experiment_id=args.experiment_id
    )

    data_module = TPPDataModule(config)

    parent_dir = "./visu"
    save_dir = os.path.join(parent_dir, args.experiment_id)

    visu = Visualizer(data_module=data_module, split="test", save_dir=save_dir)

    visu.run_visualization()


if __name__ == "__main__":
    main()
