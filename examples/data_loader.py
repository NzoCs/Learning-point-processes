import random
from typing import Any, Dict, List

from new_ltpp.configs import DataConfigBuilder
from new_ltpp.data.preprocess import TPPDataModule


def make_raw_data() -> List[List[Dict[str, Any]]]:
    data = [
        [{"time_since_last_event": 0.0, "time_since_start": 0.0, "type_event": 0}],
        [{"time_since_last_event": 0.0, "time_since_start": 0.0, "type_event": 1}],
        [{"time_since_last_event": 0.0, "time_since_start": 0.0, "type_event": 1}],
    ]
    for i, j in enumerate([2, 5, 3]):
        start_time = 0.0
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


def main_with_raw_data() -> None:
    """Example using raw data created dynamically."""
    # Create raw data
    raw_data = make_raw_data()
    print(f"📊 Created {len(raw_data)} sequences")
    for i, seq in enumerate(raw_data):
        print(f"   Sequence {i}: {len(seq)} events")

    # Build data config for raw data
    builder = DataConfigBuilder()
    builder.set_dataset_id("raw_example")
    builder.set_data_format("dict")

    builder.set_tokenizer_specs(
        {
            "num_event_types": 11,
            "pad_token_id": 11,
            "padding_side": "left",
            "max_len": 10,
        }
    )

    builder.set_data_loading_specs(
        {
            "batch_size": 2,
            "num_workers": 0,  # 0 for compatibility with raw data
            "shuffle": False,
        }
    )

    data_config = builder.build()

    # Create data module with raw data
    datamodule = TPPDataModule(data_config)
    # Note: For raw data, you might need to set the data directly
    # datamodule.raw_data = raw_data  # if supported

    print("Raw data configuration completed")
    print(
        f"Config: dataset_id={data_config.dataset_id}, batch_size={data_config.data_loading_specs.batch_size}"
    )


def main() -> None:
    # Build data config using specific builder methods
    builder = DataConfigBuilder()
    builder.set_dataset_id("example_data")
    builder.set_data_format("dict")

    # Use specific methods for data specifications
    builder.set_tokenizer_specs(
        {
            "num_event_types": 11,
            "pad_token_id": 11,
            "padding_side": "left",
            "truncation_side": "left",
        }
    )

    # Use specific method for data loading specifications
    builder.set_data_loading_specs({"batch_size": 2, "num_workers": 1, "shuffle": True})

    # Build the configuration directly
    data_config = builder.build()

    print("🔧 Data configuration created:")
    print(f"   Dataset: {data_config.dataset_id}")
    print(f"   Format: {data_config.data_format}")
    print(f"   Event types: {data_config.tokenizer_specs.num_event_types}")
    print(f"   Batch size: {data_config.data_loading_specs.batch_size}")

    # Create data module
    datamodule = TPPDataModule(data_config)

    try:
        # Setup data module
        datamodule.setup(stage="fit")

        # Get data loaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()

        print("\n✅ Data loaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

    except Exception as e:
        print(f"\n❌ Data loader creation failed: {e}")
        print("   This is expected with 'dict' format and no actual data files")


if __name__ == "__main__":
    print("=== Example 1: Raw Data Configuration ===")
    main_with_raw_data()

    print("\n=== Example 2: Standard Configuration ===")
    main()
