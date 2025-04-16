import os
import yaml
import argparse
from easy_tpp.config_factory import SimulatorConfig
from easy_tpp.simulate import Simulator

def load_config(config_path, experiment_id=None):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse the configuration with experiment_id if specified
    if experiment_id:
        print(f"Loading experiment: {experiment_id}")
        return SimulatorConfig.parse_from_yaml_config(config_dict, experiment_id=experiment_id)
    else:
        # Use top-level configuration if no experiment_id is specified
        return SimulatorConfig.parse_from_yaml_config(config_dict, direct_parse=True)

def main():
    parser = argparse.ArgumentParser(description="Run TPP simulations")
    parser.add_argument("--config", type=str, help="Path to the configuration YAML file", default="config.yaml")
    parser.add_argument("--experiment", type=str, help="Experiment ID to run (if not specified, uses default config)", default="hawkes1_simulation")
    args = parser.parse_args()
    
    # Load configuration
    simulator_config = load_config(args.config, args.experiment)
    
    # Print some configuration details
    print(f"Simulation configuration:")
    print(f"  Output directory: {simulator_config.save_dir}")
    print(f"  Time range: {simulator_config.start_time} to {simulator_config.end_time}")
    print(f"  Number of simulations: {simulator_config.num_simulations}")
    
    # Initialize and run the simulator
    simulator = Simulator(simulator_config)
    simulator.run()
    
    print(f"Simulation complete! Results saved to {simulator_config.save_dir}")

if __name__ == "__main__":
    main()
