from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "test_runner_config.yaml"

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runners import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config


def main() -> None:
    # Load configuration
    config_path = CONFIGS_DIR
    config_dict = parse_runner_yaml_config(str(config_path), "NHP", "test")
    config = RunnerConfig.from_dict(config_dict)

    # Create runner
    runner = Runner(config=config)

    # Run complete pipeline: train -> test -> predict
    print("🚀 Lancement du pipeline complet...")

    # # 1. Training
    # print("📚 Phase d'entraînement...")
    # runner.run(phase="train")

    # # 2. Testing
    # print("🧪 Phase de test...")
    # runner.run(phase="test")

    # 3. Prediction and distribution comparison
    print("🔮 Phase de prédiction et comparaison des distributions...")
    runner.run(phase="predict")


if __name__ == "__main__":
    main()
