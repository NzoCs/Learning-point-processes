from new_ltpp.configs.config_builders import DataConfigBuilder, ModelConfigBuilder
from new_ltpp.configs.data_config import DataConfig
from new_ltpp.configs.model_config import ModelConfig


def test_data_config_builder_runtime_basic():
    builder = DataConfigBuilder()
    (builder
        .set_dataset_id("test_dataset")
        .set_src_dir("/tmp/test_dataset")
        .set_data_format("json")
        .set_data_loading_specs(batch_size=16)
        .set_num_event_types(3)
    )
    # ensure required field `data_format` is provided for DataConfig
    builder.set_data_format("json")

    cfg = builder.build()
    assert isinstance(cfg, DataConfig)
    assert cfg.dataset_id == "test_dataset"
    assert cfg.train_dir == "/tmp/test_dataset"
    assert cfg.data_loading_specs.batch_size == 16
    assert cfg.tokenizer_specs.num_event_types == 3


def test_model_config_builder_runtime_basic():
    builder = ModelConfigBuilder()
    (
        builder.set_general_specs({"hidden_size": 16, "dropout": 0.1})
        .set_model_specs({})
        .set_scheduler_config(lr_scheduler=True, lr=1e-3, max_epochs=5)
        .set_simulation_config(time_window=10.0, batch_size=4, max_sim_events=100, seed=1)
        .set_thinning_config(
            num_sample=5,
            num_exp=10,
            over_sample_rate=1.2,
        )
    )

    cfg = builder.build()
    assert isinstance(cfg, ModelConfig)
    # model-level nested configs
    assert hasattr(cfg, "specs")
    assert hasattr(cfg, "simulation_config")
    assert hasattr(cfg, "thinning_config")
    assert cfg.simulation_config.batch_size == 4
    assert cfg.thinning_config.num_sample == 5
