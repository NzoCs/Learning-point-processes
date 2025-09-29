from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "yaml_configs" / "configs.yaml"

from easy_tpp.configs import RunnerConfig
from easy_tpp.runners import Runner, RunnerManager
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config
from easy_tpp.configs import ConfigType, ConfigFactory
from easy_tpp.configs.config_builder import RunnerConfigBuilder
from easy_tpp.models.model_registry import ModelRegistry



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
        logger_config_path="logger_configs.tensorboard",
    )

    config_dict = config_builder.config_dict

    config_factory = ConfigFactory()

    config = config_factory.create_config(ConfigType.RUNNER, config_dict, model_id=model_id)

    # Create runner
    runner = RunnerManager(config=config)

    # Run complete pipeline: train -> test -> predict
    print("🚀 Lancement du pipeline complet...")

    # 1. Training
    print("📚 Phase d'entraînement...")
    runner.run(phase="train")

    # 2. Testing
    print("🧪 Phase de test...")
    runner.run(phase="test")

    # 3. Prediction and distribution comparison
    print("🔮 Phase de prédiction et comparaison des distributions...")
    runner.run(phase="predict")


if __name__ == "__main__":
    main()
