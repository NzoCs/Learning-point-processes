from new_ltpp.configs.config_builders import DataConfigBuilder, RunnerConfigBuilder


def test_data_config_builder():
    """Test building a DataConfig using DataConfigBuilder."""
    builder = DataConfigBuilder()
    (
        builder.set_dataset_id("test")
        .set_src_dir("NzoCs/test_dataset")
        .set_batch_size(32)
        .set_num_workers(2)
        .set_shuffle(True)
        .set_num_event_types(2)
        
    )
    builder.set_data_format("json")

    data_config = builder.build()
    assert data_config.dataset_id == "test"
    assert data_config.train_dir == "NzoCs/test_dataset"
    assert data_config.valid_dir == "NzoCs/test_dataset"
    assert data_config.test_dir == "NzoCs/test_dataset"
    assert data_config.data_loading_specs.batch_size == 32
    assert data_config.num_event_types == 2
    assert data_config.tokenizer_specs.num_event_types == 2


def test_runner_config_builder_programmatic():
    """Test building a RunnerConfig programmatically using RunnerConfigBuilder."""
    runner_config_builder = RunnerConfigBuilder()

    # Set training config
    (
        runner_config_builder.training_builder.set_max_epochs(50)
        .set_lr(1e-3)
        .set_lr_scheduler(True)
        .set_val_freq(1)
        .set_patience(3)
        .set_accumulate_grad_batches(1)
        .set_devices(1)
    )
    # Set runner-level save dir (not a training field)
    runner_config_builder.set_save_dir("./custom_output")

    

    # Build data config
    (
        runner_config_builder.data_builder.set_num_event_types(2)
        .set_dataset_id("test")
        .set_src_dir("NzoCs/test_dataset")
        .set_batch_size(64)
        .set_num_workers(2)
        .set_shuffle(True)
        .set_data_format("json")
    )

    # Build model config
    (
        runner_config_builder.model_builder.set_general_specs({"hidden_size": 32})
        .set_model_specs({})
        .set_num_mc_samples(1)
        .set_scheduler_config(lr_scheduler=True, lr=1e-3, max_epochs=50)
        .set_simulation_config(
            time_window=30.0, batch_size=16, max_sim_events=5000, seed=42
        )
        .set_thinning_config(
            num_sample=15,
            num_exp=50,
            over_sample_rate=1.2,
        )
    )

    # Build the config
    custom_runner_config = runner_config_builder.build(model_id="NHP")

    # Assertions
    assert custom_runner_config.model_id == "NHP"
    assert custom_runner_config.data_config.dataset_id == "test"
    assert custom_runner_config.training_config.max_epochs == 50
    assert "custom_output" in custom_runner_config.save_dir
    assert custom_runner_config.model_config.specs.hidden_size == 32


def test_runner_config_builder_from_yaml(tmp_path):
    """Test loading RunnerConfig from YAML using RunnerConfigBuilder."""
    # Create a temporary YAML config file
    yaml_content = """
training_configs:
  quick_test:
    max_epochs: 10
    lr: 0.001
    lr_scheduler: true
    val_freq: 1
    patience: 3

model_configs:
  neural_small:
    general_specs:
      hidden_size: 32
    model_specs: {}

data_configs:
  test:
    dataset_id: test
    src_dir: NzoCs/test_dataset
    num_event_types: 2

data_loading_configs:
  quick_test:
    batch_size: 32
    num_workers: 2
    shuffle: true

thinning_configs:
  thinning_fast:
    num_sample: 10
    num_exp: 50
    num_samples_boundary: 5

simulation_configs:
  simulation_fast:
    time_window: 30.0
    batch_size: 16
    initial_buffer_size: 1000
    seed: 42

logger_configs:
  csv:
    type: csv
    save_dir: ./logs
    name: test_experiment
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    # Build config from YAML using the YAML loaders and then populate the builder
    from new_ltpp.configs.config_loaders.runner_config_loader import (
      RunnerConfigYamlLoader,
    )

    runner_builder = RunnerConfigBuilder()
    loader = RunnerConfigYamlLoader()
    runner_cfg = loader.load(
      str(yaml_file),
      training_config_path="training_configs.quick_test",
      model_config_path="model_configs.neural_small",
      data_config_path="data_configs.test",
      data_loading_config_path="data_loading_configs.quick_test",
      thinning_config_path="thinning_configs.thinning_fast",
      simulation_config_path="simulation_configs.simulation_fast",
      logger_config_path="logger_configs.csv",
    )

    # Populate builder and check for missing required fields
    runner_builder.from_dict(runner_cfg)
    # Explicitly set required fields omitted by YAML so tests control these values
    runner_builder.model_builder.set_num_mc_samples(1)
    runner_builder.data_builder.set_data_format("json")
    missing = (
      runner_builder.model_builder.get_unset_required_fields()
      + runner_builder.data_builder.get_unset_required_fields()
      + runner_builder.training_builder.get_unset_required_fields()
    )
    assert len(missing) == 0  # No missing fields

    runner_config = runner_builder.build(model_id="NHP")

    # Assertions
    assert runner_config.model_id == "NHP"
    assert runner_config.training_config.max_epochs == 10
    assert runner_config.data_config.dataset_id == "test"
