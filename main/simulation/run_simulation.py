import os
import yaml
import argparse
from easy_tpp.config_factory import Config
from easy_tpp.simulate import Simulator


def main():

    parser = argparse.ArgumentParser(description="Run TPP simulations")
    parser.add_argument("--config_dir", type=str, help="Path to the configuration YAML file", default="simul_config.yaml")
    parser.add_argument("--experiment_id", type=str, help="Experiment ID to run (if not specified, uses default config)", default="AttNHP")
    parser .add_argument("--dataset_id", type=str, help="Dataset id on which the model was trained", default="hawkes1")
    args = parser.parse_args()
    
    # Load configuration
    simulator_config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id, dataset_id=args.dataset_id)

    # Initialize and run the simulator
    simulator = Simulator(simulator_config)
    simulator.run()

if __name__ == "__main__":
    main()