from new_ltpp.configs.config_builders import DataConfigBuilder, RunnerConfigBuilder


def test_data_config_builder():
    """Test building a DataConfig using DataConfigBuilder."""
    builder = DataConfigBuilder()
    builder.set_dataset_id("test")
    builder.set_src_dir("NzoCs/test_dataset")
    builder.set_batch_size(32)
    builder.set_num_workers(2)
    builder.set_shuffle(True)
    builder.set_num_event_types(2)
    builder.set_max_len(128)

    data_config = builder.build()
    assert data_config.dataset_id == "test"
    assert data_config.train_dir == "NzoCs/test_dataset"
    assert data_config.valid_dir == "NzoCs/test_dataset"
    assert data_config.test_dir == "NzoCs/test_dataset"
    assert data_config.data_loading_specs.batch_size == 32
    assert data_config.num_event_types == 2
    assert data_config.tokenizer_specs.max_len == 128


def test_runner_config_builder_programmatic():
    """Test building a RunnerConfig programmatically using RunnerConfigBuilder."""
    runner_config_builder = RunnerConfigBuilder()

    # Set training config
    runner_config_builder.training_builder.set_max_epochs(50)
    runner_config_builder.training_builder.set_batch_size(64)
    runner_config_builder.training_builder.set_lr(1e-3)
    runner_config_builder.training_builder.set_lr_scheduler(True)
    runner_config_builder.training_builder.set_val_freq(1)
    runner_config_builder.training_builder.set_patience(3)
    runner_config_builder.training_builder.set_accumulate_grad_batches(1)
    runner_config_builder.training_builder.set_devices(1)
    runner_config_builder.set_save_dir("./custom_output")

    # Build data config
    runner_config_builder.data_builder.set_num_event_types(2)
    runner_config_builder.data_builder.set_dataset_id("test")
    runner_config_builder.data_builder.set_src_dir("NzoCs/test_dataset")
    runner_config_builder.data_builder.set_batch_size(64)
    runner_config_builder.data_builder.set_num_workers(2)
    runner_config_builder.data_builder.set_shuffle(True)
    runner_config_builder.data_builder.set_max_len(128)

    # Build model config
    runner_config_builder.model_builder.set_general_specs({"hidden_size": 32})
    runner_config_builder.model_builder.set_model_specs({})
    runner_config_builder.model_builder.set_scheduler_config(lr_scheduler=True, lr=1e-3)
    runner_config_builder.model_builder.set_simulation_config(
        start_time=20, end_time=50, batch_size=16, max_sim_events=5000, seed=42
    )
    runner_config_builder.model_builder.set_thinning_config(
        num_sample=15,
        num_exp=50,
        use_mc_samples=True,
        loss_integral_num_sample_per_step=10,
        num_steps=10,
        over_sample_rate=1.2,
        num_samples_boundary=5,
        dtime_max=3.0,
    )

    # Build the config
    custom_runner_config = runner_config_builder.build(model_id="NHP")

    # Assertions
    assert custom_runner_config.model_id == "NHP"
    assert custom_runner_config.data_config.dataset_id == "test"
    assert custom_runner_config.training_config.max_epochs == 50
    assert custom_runner_config.training_config.batch_size == 64
    assert "custom_output" in custom_runner_config.save_dir
    assert custom_runner_config.model_config.specs["hidden_size"] == 32


def test_runner_config_builder_from_yaml(tmp_path):
    """Test loading RunnerConfig from YAML using RunnerConfigBuilder."""
    # Create a temporary YAML config file
    yaml_content = """
training_configs:
  quick_test:
    max_epochs: 10
    batch_size: 32
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

simulation_configs:
  simulation_fast:
    start_time: 0
    end_time: 50
    batch_size: 16
    max_sim_events: 1000
    seed: 42

logger_configs:
  csv:
    type: csv
    save_dir: ./logs
    name: test_experiment
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    # Build config from YAML
    runner_builder = RunnerConfigBuilder()
    missing = runner_builder.load_from_yaml(
        yaml_file_path=str(yaml_file),
        training_config_path="training_configs.quick_test",
        model_config_path="model_configs.neural_small",
        data_config_path="data_configs.test",
        data_loading_config_path="data_loading_configs.quick_test",
        thinning_config_path="thinning_configs.thinning_fast",
        simulation_config_path="simulation_configs.simulation_fast",
        logger_config_path="logger_configs.csv",
    )

    assert len(missing) == 0  # No missing fields

    runner_config = runner_builder.build(model_id="NHP")

    # Assertions
    assert runner_config.model_id == "NHP"
    assert runner_config.training_config.max_epochs == 10
    assert runner_config.data_config.dataset_id == "test"
