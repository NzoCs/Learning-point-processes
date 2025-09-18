from easy_tpp.config_factory import DataConfig
from easy_tpp.data.preprocess import TPPDataModule
from easy_tpp.data.preprocess.visualizer import Visualizer


def inspect_dataset() -> None:
    """Inspect dataset distribution and statistics."""
    # Configuration for data inspection
    data_config = DataConfig(
        dataset_id="test", data_format="pickle", num_event_types=2, batch_size=32
    )

    # Create data module
    datamodule = TPPDataModule(data_config)
    datamodule.setup()

    # Create visualizer
    visualizer = Visualizer(
        data_setup=datamodule, split="train", save_dir="./inspection_plots"
    )

    # Generate analysis plots
    visualizer.show_all_distributions(save_graph=True, show_graph=False)
    visualizer.delta_times_distribution(save_graph=True)
    visualizer.event_type_distribution(save_graph=True)

    print("Data inspection completed - check ./inspection_plots")


def inspect_synthetic_data() -> None:
    """Inspect synthetic data generated from gen_synthetic_data.py."""
    data_config = DataConfig(
        dataset_id="synthetic_hawkes_data",
        data_format="json",
        num_event_types=3,
        batch_size=16,
    )

    datamodule = TPPDataModule(data_config)
    datamodule.setup()

    visualizer = Visualizer(
        data_setup=datamodule, split="train", save_dir="./synthetic_inspection_plots"
    )

    visualizer.show_all_distributions(save_graph=True, show_graph=False)
    print("Synthetic data inspection completed - check ./synthetic_inspection_plots")


def main() -> None:
    inspect_dataset()
    inspect_synthetic_data()


if __name__ == "__main__":
    main()
