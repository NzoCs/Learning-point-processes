import random
from typing import Any, Dict, List

from easy_tpp.configs.config_builder import DataConfigBuilder
from easy_tpp.configs.config_factory import ConfigFactory, ConfigType
from easy_tpp.data.preprocess import TPPDataModule


def make_raw_data() -> List[List[Dict[str, Any]]]:
    data = [
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 0}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1}],
    ]
    for i, j in enumerate([2, 5, 3]):
        start_time = 0
        for k in range(j):
            delta_t = random.random()
            start_time += delta_t
            data[i].append(
                {
                    "time_since_last_event": delta_t,
                    "time_since_start": start_time,
                    "type_event": random.randint(0, 10),
                }
            )
    return data


def main() -> None:
    # Build data config using the builder + factory pattern
    builder = DataConfigBuilder()
    builder.set_field("dataset_id", "example_data")
    builder.set_field("data_format", "dict")
    builder.set_field("num_event_types", 11)
    builder.set_field("pad_token_id", 11)
    builder.set_field("batch_size", 2)

    config_dict = builder.get_config_dict()
    config_factory = ConfigFactory()
    data_config = config_factory.create_config(ConfigType.DATA, config_dict)

    # Create data module
    datamodule = TPPDataModule(data_config)

    # Setup data module
    datamodule.setup(stage="fit")

    # Get data loaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    print("Train loader created successfully")
    print(f"Number of batches in train loader: {len(train_loader)}")


if __name__ == "__main__":
    main()
