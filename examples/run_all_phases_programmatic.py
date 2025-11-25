from pathlib import Path

from new_ltpp.configs import ConfigFactory, ConfigType
from new_ltpp.configs.config_builders import RunnerConfigBuilder
from new_ltpp.runners import RunnerManager


def main() -> None:
    """
    Run complete pipeline (train -> test -> predict) using programmatic configuration
    without YAML files - only using set_* methods.
    """
    print("⚙️ Building complete runner config programmatically without YAML...")

    # Create a new RunnerConfigBuilder instance
    builder = RunnerConfigBuilder()

    # Build data config using the data_builder attribute
    print("📊 Configuring data settings...")
    builder.data_builder.set_num_event_types(2)
    builder.data_builder.set_dataset_id("test")
    builder.data_builder.set_src_dir("NzoCs/test_dataset")
    builder.data_builder.set_batch_size(64)
    builder.data_builder.set_num_workers(2)
    builder.data_builder.set_shuffle(True)
    builder.data_builder.set_max_len(128)

    # Build model config using the model_builder attribute
    print("🧠 Configuring model settings...")
    builder.model_builder.set_general_specs({"hidden_size": 64})
    builder.model_builder.set_model_specs({})

    # Set scheduler config with explicit parameters
    builder.model_builder.set_scheduler_config(lr_scheduler=True, lr=1e-3)

    # Set simulation config with explicit parameters
    builder.model_builder.set_simulation_config(
        start_time=20, end_time=50, batch_size=16, max_sim_events=500, seed=42
    )

    # Set thinning config with explicit parameters
    builder.model_builder.set_thinning_config(
        num_sample=20,
        num_exp=100,
        use_mc_samples=True,
        loss_integral_num_sample_per_step=10,
        num_steps=10,
        over_sample_rate=1.2,
        num_samples_boundary=5,
        dtime_max=3.0,
    )

    # Set training config using convenience methods
    print("🏋️ Configuring training settings...")
    builder.set_max_epochs(5)
    builder.set_batch_size(64)
    builder.set_lr(1e-3)
    builder.set_lr_scheduler(True)
    builder.set_val_freq(5)
    builder.set_patience(10)
    builder.set_accumulate_grad_batches(1)

    # Set runner-specific parameters
    builder.set_save_dir("./programmatic_output")

    # Build the final config
    model_id = "NHP"
    config = builder.build(model_id=model_id)

    print("✅ Programmatic runner config built successfully!")
    print(f"   🧠 Model: {model_id}")
    print(f"   📊 Dataset: test")
    print(f"   🔢 Max Epochs: {config.training_config.max_epochs}")
    print(f"   📦 Batch Size: {config.training_config.batch_size}")
    print(f"   💾 Save Dir: {config.save_dir}")

    # Create runner
    print("\n🚀 Initializing runner...")
    runner = RunnerManager(config=config)

    # Run complete pipeline: train -> test -> predict
    print("\n" + "=" * 60)
    print("🏃 STARTING COMPLETE PIPELINE")
    print("=" * 60)

    try:
        # 1. Training phase
        print("\n🏋️ Phase 1: Training")
        print("-" * 30)
        runner.run(phase="train")
        print("✅ Training completed successfully!")

        # 2. Testing phase
        print("\n🧪 Phase 2: Testing")
        print("-" * 30)
        runner.run(phase="test")
        print("✅ Testing completed successfully!")

        # 3. Prediction and distribution comparison phase
        print("\n🔮 Phase 3: Prediction and Distribution Comparison")
        print("-" * 50)
        runner.run(phase="predict")
        print("✅ Prediction completed successfully!")

        print("\n" + "=" * 60)
        print("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Results saved in: {config.save_dir}")

    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
