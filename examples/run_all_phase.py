from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "yaml_configs" / "configs.yaml"

from easy_tpp.runners import RunnerManager
from easy_tpp.configs import ConfigType, ConfigFactory
from easy_tpp.configs.config_builder import RunnerConfigBuilder



def main() -> None:
    # Load configuration
    config_path = CONFIGS_DIR
    model_id = "NHP"

    # Build runner configuration from YAML
    config_builder = RunnerConfigBuilder()

    # You can modify the paths below to point to different configurations as needed
    config_builder.load_from_yaml(
        yaml_file_path=config_path,
        data_config_path="data_configs.test",
        training_config_path="training_configs.quick_test",
        model_config_path="model_configs.neural_small",
        thinning_config_path="thinning_configs.thinning_fast",
        simulation_config_path="simulation_configs.simulation_fast",
        data_loading_config_path="data_loading_configs.quick_test",
        logger_config_path="logger_configs.mlflow",
    )

    config = config_builder.build()

    # Create runner
    runner = RunnerManager(config=config)

    # Run complete pipeline: train -> test -> predict

    # 1. Training
    runner.run(phase="train")

    # 2. Testing
    runner.run(phase="test")

    # 3. Prediction and distribution comparison
    runner.run(phase="predict")


if __name__ == "__main__":
    main()
