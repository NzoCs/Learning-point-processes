from pathlib import Path
import yaml
import pytest

from new_ltpp.configs.config_loaders.runner_config_loader import RunnerConfigYamlLoader
from new_ltpp.configs.config_builders import RunnerConfigBuilder


def make_sample_yaml(tmp_path: Path) -> Path:
    data = {
        "training_configs": {"quick": {"max_epochs": 1, "batch_size": 2, "lr": 0.01, "lr_scheduler": True}},
        "model_configs": {
            "m1": {"general_specs": {"hidden_size": 8}, "model_specs": {}, "num_mc_samples": 1}
        },
        "data_configs": {"d1": {"dataset_id": "d1", "src_dir": "./data", "num_event_types": 2, "data_format": "json"}},
        "data_loading_configs": {"dl1": {"batch_size": 2, "num_workers": 1}},
        "thinning_configs": {"t1": {"num_sample": 5, "num_exp": 10}},
        "simulation_configs": {"s1": {"time_window": 10, "batch_size": 2, "initial_buffer_size": 100}},
        "logger_configs": {"csv": {"type": "csv", "save_dir": "./logs"}},
    }
    p = tmp_path / "runner_test.yaml"
    p.write_text(yaml.dump(data))
    return p


def test_runner_loader_raises_when_required_subblocks_missing(tmp_path: Path):
    yaml_path = make_sample_yaml(tmp_path)
    loader = RunnerConfigYamlLoader()

    # missing model_config_path (not provided) should raise TypeError (required kwarg)
    with pytest.raises(TypeError):
        loader.load(
            str(yaml_path),
            training_config_path="training_configs.quick",
            data_config_path="data_configs.d1",
            data_loading_config_path="data_loading_configs.dl1",
            thinning_config_path="thinning_configs.t1",
            simulation_config_path="simulation_configs.s1",
            logger_config_path="logger_configs.csv",
        ) # type: ignore

    # missing data_loading_config_path (not provided) should raise TypeError (required kwarg)
    with pytest.raises(TypeError):
        loader.load(
            str(yaml_path),
            training_config_path="training_configs.quick",
            data_config_path="data_configs.d1",
            model_config_path="model_configs.m1",
            thinning_config_path="thinning_configs.t1",
            simulation_config_path="simulation_configs.s1",
            logger_config_path="logger_configs.csv",
        ) # type: ignore


def test_runner_loader_success_with_required_paths(tmp_path: Path):
    yaml_path = make_sample_yaml(tmp_path)
    loader = RunnerConfigYamlLoader()

    cfg = loader.load(
        str(yaml_path),
        training_config_path="training_configs.quick",
        data_config_path="data_configs.d1",
        model_config_path="model_configs.m1",
        data_loading_config_path="data_loading_configs.dl1",
        thinning_config_path="thinning_configs.t1",
        simulation_config_path="simulation_configs.s1",
        logger_config_path="logger_configs.csv",
    )
    # Load into builder and verify sub-builders are populated
    runner_builder = RunnerConfigBuilder()
    runner_builder.from_dict(cfg)

    # Required fields are provided in the YAML; no manual setters needed

    missing = (
        runner_builder.model_builder.get_unset_required_fields()
        + runner_builder.data_builder.get_unset_required_fields()
        + runner_builder.training_builder.get_unset_required_fields()
    )
    assert len(missing) == 0
    assert "training_config" in cfg
    assert "model_config" in cfg
    assert "data_config" in cfg
    assert cfg["model_config"].get("general_specs") is not None
