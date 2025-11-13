from pathlib import Path

from new_ltpp.globals import CONFIGS_FILE
from new_ltpp.configs.config_builders import RunnerConfigBuilder
from new_ltpp.runners import RunnerManager


def main() -> None:
    # Load configuration
    config_path = CONFIGS_FILE
    model_id = "NHP"

    # Build runner configuration from YAML
    config_builder = RunnerConfigBuilder()

    # You can modify the paths below to point to different configurations as needed
    config_builder.load_from_yaml(
        yaml_file_path=config_path,
        data_config_path="data_configs.test",
        training_config_path="training_configs.quick_test",
        model_config_path="model_configs.quick_test",
        thinning_config_path="thinning_configs.quick_test",
        simulation_config_path="simulation_configs.quick_test",
        data_loading_config_path="data_loading_configs.quick_test",
        logger_config_path="logger_configs.csv",
    )

    config = config_builder.build(model_id=model_id)

    # Create runner
    runner = RunnerManager(config=config)

    # Run complete pipeline: train -> test -> predict

    # # 1. Training
    # runner.run(phase="train")

    # # 2. Testing
    # runner.run(phase="test")

    # 3. Prediction and distribution comparison
    runner.run(phase="predict")


if __name__ == "__main__":
    main()
