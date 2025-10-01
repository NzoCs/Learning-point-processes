from easy_tpp.configs import DataConfigBuilder
from easy_tpp.data.preprocess import TPPDataModule
from easy_tpp.data.preprocess.visualizer import Visualizer


def inspect_dataset() -> None:
    """Inspect dataset distribution and statistics."""
    # Configuration for data inspection (use builder)
    builder = DataConfigBuilder()
    builder.set_field("dataset_id", "test")
    builder.set_field("train_dir", "NzoCs/test_dataset")
    builder.set_field("valid_dir", "NzoCs/test_dataset")
    builder.set_field("test_dir", "NzoCs/test_dataset")
    builder.set_field("data_loading_specs", {"batch_size": 32})
    builder.set_field("data_specs", {"num_event_types": 2})
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



def main() -> None:
    inspect_dataset()

if __name__ == "__main__":
    main()
