import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Importing config builders and data module...")
    from new_ltpp.configs.config_builders import DataConfigBuilder
    from new_ltpp.data.preprocess import TPPDataModule

    print("Creating DataConfigBuilder...")
    data_builder = DataConfigBuilder()
    (
        data_builder.set_dataset_id("test")
        .set_src_dir("NzoCs/test_dataset")
        .set_num_event_types(1)
        .set_data_loading_specs(batch_size=16, num_workers=0, shuffle=False)
        .set_data_format("hf")
    )
    data_config = data_builder.build()
    print("Data config built:", data_config)

    print("Creating TPPDataModule...")
    datamodule = TPPDataModule(data_config)
    print("TPPDataModule created:", datamodule)

    print("Setting up datamodule (test stage)...")
    datamodule.setup(stage="test")
    print("Datamodule setup done!")

    test_loader = datamodule.test_dataloader()
    print("Test loader created, length:", len(test_loader))

    batch = next(iter(test_loader))
    print("Batch loaded successfully. Shape:", batch.time_seqs.shape)

except Exception as e:
    import traceback

    traceback.print_exc()
