"""
Unit tests for the refactored TrainerConfig and RunnerConfig.
"""

import pytest
from easy_tpp.configs.runner_config import TrainerConfig, RunnerConfig
from easy_tpp.configs.model_config import ModelConfig
from easy_tpp.configs.data_config import DataConfig
from easy_tpp.configs.logger_config import LoggerConfig


def test_trainer_config_defaults(tmp_path):
    config = TrainerConfig(dataset_id="ds1", model_id="m1", save_dir=str(tmp_path))
    assert config.dataset_id == "ds1"
    assert config.model_id == "m1"
    assert config.batch_size == 32
    assert config.save_model_dir.endswith("trained_models")
    assert config.devices in [-1, 0, 1]  # Accepts CPU or CUDA
    assert config.logger_config.save_dir == str(tmp_path)


def test_trainer_config_from_dict(tmp_path):
    d = {
        "dataset_id": "ds2",
        "model_id": "m2",
        "batch_size": 64,
        "logger_config": {"log_level": "DEBUG"},
        "save_dir": str(tmp_path),
    }
    config = TrainerConfig.from_dict(d)
    assert config.dataset_id == "ds2"
    assert config.batch_size == 64
    assert config.logger_config.log_level == "DEBUG"


def test_trainer_config_required_fields():
    with pytest.raises(TypeError):
        TrainerConfig(batch_size=32)  # Missing required fields


def test_runner_config_from_dict(tmp_path):
    trainer = TrainerConfig(dataset_id="ds3", model_id="m3", save_dir=str(tmp_path))
    model = ModelConfig(model_id="NHP", num_event_types=5)
    data = DataConfig(train_dir="train.csv", data_specs={"num_event_types": 5})
    d = {"trainer_config": trainer, "model_config": model, "data_config": data}
    config = RunnerConfig.from_dict(d)
    assert isinstance(config.trainer_config, TrainerConfig)
    assert isinstance(config.model_config, ModelConfig)
    assert isinstance(config.data_config, DataConfig)


def test_runner_config_yaml():
    trainer = TrainerConfig(dataset_id="ds4", model_id="m4")
    model = ModelConfig(model_id="NHP", num_event_types=5)
    data = DataConfig(train_dir="train.csv", data_specs={"num_event_types": 5})
    config = RunnerConfig(trainer_config=trainer, model_config=model, data_config=data)
    yaml_dict = config.get_yaml_config()
    assert "trainer_config" in yaml_dict
    assert "model_config" in yaml_dict
    assert "data_config" in yaml_dict
