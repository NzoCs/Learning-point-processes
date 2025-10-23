from new_ltpp.configs import DataConfigBuilder
from new_ltpp.data.preprocess import TPPDataModule
from new_ltpp.data.preprocess.visualizer import Visualizer


def inspect_dataset() -> None:
    """Inspect dataset distribution and statistics."""
    # Configuration for data inspection (use specific builder methods)
    # Available methods: set_src_dir(), set_train_dir(), set_valid_dir(), set_test_dir(),
    # set_data_loading_specs(), set_data_specs(), set_field() for other fields
    builder = DataConfigBuilder()
    builder.set_dataset_id("test")
    builder.set_src_dir("NzoCs/test_dataset")
    builder.set_data_loading_specs({"batch_size": 32})
    builder.set_tokenizer_specs({"num_event_types": 2})
    data_config = builder.build()

    # Create data module
    datamodule = TPPDataModule(data_config)
    datamodule.setup(stage="test")

    # Create visualizer
    visualizer = Visualizer(
        data_module=datamodule, split="train", save_dir="./inspection_plots"
    )

    # Generate analysis plots
    visualizer.show_all_distributions(save_graph=True, show_graph=False)
    visualizer.delta_times_distribution(save_graph=True)
    visualizer.event_type_distribution(save_graph=True)

    print("Data inspection completed - check ./inspection_plots")


def inspect_dataset_from_yaml() -> None:
    """Inspect dataset using configuration loaded from YAML file."""
    # Configuration for data inspection (load from YAML)
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path="../yaml_configs/configs.yaml",  # Path to your YAML config file
        data_config_path="data_configs.test",  # Path to data config section
        data_loading_config_path="data_loading_configs.quick_test",  # Path to data loading config
        data_specs_path="tokenizer_specs.standard",  # Optional: path to data specs
    )
    data_config = builder.build()

    print(f"📊 Loaded data configuration from YAML:")
    print(f"   Dataset: {data_config.dataset_id}")
    print(f"   Format: {data_config.data_format}")
    print(f"   Train dir: {data_config.train_dir}")

    # Create data module
    datamodule = TPPDataModule(data_config)
    datamodule.setup(stage="test")

    # Create visualizer with YAML-loaded config
    visualizer = Visualizer(
        data_module=datamodule, split="train", save_dir="./yaml_inspection_plots"
    )

    # Generate analysis plots
    visualizer.show_all_distributions(save_graph=True, show_graph=False)
    visualizer.delta_times_distribution(save_graph=True, show_graph=False)
    visualizer.event_type_distribution(save_graph=True, show_graph=False)
    visualizer.sequence_length_distribution(save_graph=True, show_graph=False)

    print("YAML-based data inspection completed - check ./yaml_inspection_plots")


def inspect_dataset_custom_yaml() -> None:
    """Inspect dataset with custom YAML configuration paths."""
    # Configuration with different YAML paths
    builder = DataConfigBuilder()
    builder.load_from_yaml(
        yaml_path="../yaml_configs/configs.yaml",
        data_config_path="data_configs.retweets",  # Different dataset
        data_loading_config_path="data_loading_configs.default",  # Different loading config
    )
    data_config = builder.build()


def inspect_dataset_manual_methods() -> None:
    """Inspect dataset using individual builder methods."""
    # Configuration using specific methods instead of set_field
    builder = DataConfigBuilder()

    # Use specific methods for each configuration aspect
    builder.set_dataset_id("test")
    builder.set_train_dir("data/train")
    builder.set_valid_dir("data/valid")
    builder.set_test_dir("data/test")
    # Alternative: use set_src_dir() to set all three at once
    # builder.set_src_dir("data/combined")

    builder.set_data_loading_specs(
        {"batch_size": 64, "num_workers": 4, "shuffle": True}
    )

    builder.set_tokenizer_specs(
        {
            "num_event_types": 5,
            "padding_side": "left",
            "truncation_side": "right",
            "max_len": 256,
        }
    )

    data_config = builder.build()

    print(f"🔧 Manual methods configuration:")
    print(f"   Dataset: {data_config.dataset_id}")
    print(f"   Train dir: {data_config.train_dir}")
    print(f"   Valid dir: {data_config.valid_dir}")
    print(f"   Test dir: {data_config.test_dir}")
    print(f"   Batch size: {data_config.data_loading_specs.batch_size}")
    print(f"   Event types: {data_config.tokenizer_specs.num_event_types}")

    try:
        # Create data module
        datamodule = TPPDataModule(data_config)
        datamodule.setup(stage="test")

        # Create visualizer
        visualizer = Visualizer(
            data_module=datamodule, split="train", save_dir="./manual_methods_plots"
        )

        # Generate plots
        visualizer.event_type_distribution(save_graph=True, show_graph=False)

        print("Manual methods inspection completed - check ./manual_methods_plots")

    except Exception as e:
        print(f"Manual methods example failed (expected with dummy paths): {e}")

    print(f"🔧 Custom YAML configuration loaded:")
    print(f"   Dataset: {data_config.dataset_id}")
    print(f"   Batch size: {data_config.data_loading_specs.batch_size}")
    print(f"   Event types: {data_config.tokenizer_specs.num_event_types}")

    # Create data module
    datamodule = TPPDataModule(data_config)
    datamodule.setup(stage="test")

    # Create visualizer
    visualizer = Visualizer(
        data_module=datamodule,
        split="train",
        save_dir="./custom_yaml_plots",
        dataset_size=5000,  # Limit dataset size for faster processing
    )

    # Generate specific plots
    visualizer.delta_times_distribution(save_graph=True, show_graph=False)
    visualizer.event_type_distribution(save_graph=True, show_graph=False)

    print("Custom YAML inspection completed - check ./custom_yaml_plots")


def main() -> None:
    """Main function to run different inspection examples."""
    print("=== Example 1: Manual Configuration ===")
    inspect_dataset()

    print("\n=== Example 2: YAML Configuration ===")
    try:
        inspect_dataset_from_yaml()
    except Exception as e:
        print(f"YAML example failed: {e}")

    print("\n=== Example 3: Custom YAML Paths ===")
    try:
        inspect_dataset_custom_yaml()
    except Exception as e:
        print(f"Custom YAML example failed: {e}")

    print("\n=== Example 4: Individual Builder Methods ===")
    inspect_dataset_manual_methods()


if __name__ == "__main__":
    main()
