"""Runner-based integration tests for the complete EasyTPP system.

This module provides integration tests that use the runner configuration approach,
similar to how models are instantiated in the main training scripts.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

# from easy_tpp.config_factory import Config
from easy_tpp.configs import RunnerConfig
from easy_tpp.runners import Trainer
from easy_tpp.utils.torch_utils import set_seed


@pytest.mark.integration
class TestRunnerBasedIntegration:
    """Integration tests using the runner-based configuration approach."""

    def setup_method(self):
        """Set up for each test method."""
        set_seed(42)
        self.test_configs = self._create_test_configs()

    def _create_test_configs(self):
        """Create test configuration dictionaries for different models."""
        base_config = {
            "pipeline_config_id": "runner_config",
            "data": {
                "test_data": {
                    "data_format": "json",
                    "train_dir": "NzoCs/test-temporal-dataset",
                    "valid_dir": "NzoCs/test-temporal-dataset",
                    "test_dir": "NzoCs/test-temporal-dataset",
                    "data_specs": {
                        "num_event_types": 2,
                        "pad_token_id": 2,
                        "padding_side": "left",
                    },
                }
            },
        }

        # Configuration for NHP model
        nhp_config = {
            **base_config,
            "NHP_test": {
                "data_loading_specs": {"batch_size": 8, "num_workers": 1},
                "model_config": {
                    "model_id": "NHP",
                    "specs": {"hidden_size": 32, "time_emb_size": 8, "num_layers": 1},
                    "thinning": {
                        "num_exp": 50,
                        "over_sample_rate": 1.5,
                        "dtime_max": 5,
                        "num_sample": 10,
                    },
                    "base_config": {"lr": 0.01, "lr_scheduler": False},
                    "simulation_config": {
                        "start_time": 10,
                        "end_time": 20,
                        "max_sim_events": 100,
                    },
                },
                "trainer_config": {
                    "stage": "train",
                    "max_epochs": 2,
                    "val_freq": 1,
                    "accumulate_grad_batches": 1,
                    "patience": 5,
                    "logger_config": {"type": "tensorboard", "name": "test_logs"},
                    "save_model_dir": "test_checkpoints",
                    "devices": 1,
                    "use_precision_16": False,
                    "log_freq": 1,
                    "checkpoints_freq": 1,
                },
            },
        }

        # Configuration for RMTPP model
        rmtpp_config = {
            **base_config,
            "RMTPP_test": {
                "data_loading_specs": {"batch_size": 8, "num_workers": 1},
                "model_config": {
                    "model_id": "RMTPP",
                    "specs": {"hidden_size": 16, "time_emb_size": 8, "num_layers": 1},
                    "thinning": {
                        "num_exp": 50,
                        "over_sample_rate": 1.5,
                        "dtime_max": 5,
                        "num_sample": 10,
                    },
                    "base_config": {"lr": 0.01, "lr_scheduler": False},
                    "simulation_config": {
                        "start_time": 10,
                        "end_time": 20,
                        "max_sim_events": 100,
                    },
                },
                "trainer_config": {
                    "stage": "train",
                    "max_epochs": 2,
                    "val_freq": 1,
                    "accumulate_grad_batches": 1,
                    "patience": 5,
                    "logger_config": {"type": "tensorboard", "name": "test_logs"},
                    "save_model_dir": "test_checkpoints",
                    "devices": 1,
                    "use_precision_16": False,
                    "log_freq": 1,
                    "checkpoints_freq": 1,
                },
            },
        }

        # Configuration for Hawkes model
        hawkes_config = {
            **base_config,
            "Hawkes_test": {
                "data_loading_specs": {"batch_size": 8, "num_workers": 1},
                "model_config": {
                    "model_id": "HawkesModel",
                    "specs": {
                        "mu": [0.1, 0.1],
                        "alpha": [[0.2, 0], [0, 0.4]],
                        "beta": [[1.0, 0], [0, 2.0]],
                    },
                    "thinning": {
                        "num_exp": 50,
                        "num_sample": 10,
                        "over_sample_rate": 1.5,
                        "dtime_max": 5,
                    },
                },
                "trainer_config": {
                    "stage": "test",
                    "save_model_dir": "test_checkpoints",
                    "devices": 1,
                    "use_precision_16": False,
                    "log_freq": 1,
                    "checkpoints_freq": 1,
                },
            },
        }

        return {"NHP": nhp_config, "RMTPP": rmtpp_config, "Hawkes": hawkes_config}

    def _create_temp_config_file(self, config_dict, temp_dir, experiment_id=None):
        """Create a temporary YAML config file, flattening experiment-specific config."""
        config_path = temp_dir / "test_config.yaml"
        # If experiment_id is provided and present in config_dict, flatten it
        if experiment_id and experiment_id in config_dict:
            flat_config = dict(config_dict)  # shallow copy
            exp_cfg = flat_config.pop(experiment_id)
            flat_config.update(exp_cfg)
        else:
            flat_config = config_dict
        with open(config_path, "w") as f:
            yaml.dump(flat_config, f, default_flow_style=False)
        return str(config_path)

    @pytest.mark.parametrize("model_name", ["NHP", "RMTPP"])
    def test_runner_model_instantiation(self, temporary_directory, model_name):
        """Test model instantiation through runner configuration."""
        config_dict = self.test_configs[model_name]
        config_path = self._create_temp_config_file(config_dict, temporary_directory)

        # Get experiment name based on model
        experiment_id = f"{model_name}_test"

        # Mock the data loading to avoid file system dependencies
        with patch("easy_tpp.preprocess.data_loader.TPPDataModule.setup"):
            with patch(
                "easy_tpp.preprocess.data_loader.TPPDataModule.train_dataloader"
            ):
                with patch(
                    "easy_tpp.preprocess.data_loader.TPPDataModule.val_dataloader"
                ):
                    with patch(
                        "easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader"
                    ):
                        # Build config from YAML
                        config = RunnerConfig.load_from_yaml_file(config_path)
                        # Set experiment_id and dataset_id if needed
                        config.experiment_id = experiment_id
                        config.dataset_id = "test_data"
                        # Create trainer instance
                        trainer = Trainer(config, output_dir=str(temporary_directory))

                        # Verify trainer was created successfully
                        assert trainer is not None
                        assert trainer.model is not None
                        assert trainer.model_id == model_name
                        assert trainer.datamodule is not None

    @pytest.mark.parametrize("model_name", ["NHP", "RMTPP"])
    def test_runner_training_step(self, temporary_directory, model_name):
        """Test training step through runner."""
        config_dict = self.test_configs[model_name]
        config_path = self._create_temp_config_file(config_dict, temporary_directory)

        experiment_id = f"{model_name}_test"

        # Mock data loaders
        mock_train_loader = Mock()
        mock_val_loader = Mock()

        with patch("easy_tpp.preprocess.data_loader.TPPDataModule.setup"):
            with patch(
                "easy_tpp.preprocess.data_loader.TPPDataModule.train_dataloader",
                return_value=mock_train_loader,
            ):
                with patch(
                    "easy_tpp.preprocess.data_loader.TPPDataModule.val_dataloader",
                    return_value=mock_val_loader,
                ):
                    with patch("pytorch_lightning.Trainer.fit") as mock_fit:
                        # Build config and create trainer
                        config = RunnerConfig.load_from_yaml_file(config_path)
                        config.experiment_id = experiment_id
                        config.dataset_id = "test_data"
                        trainer = Trainer(config, output_dir=str(temporary_directory))

                        # Test training
                        trainer.train()

                        # Verify training was called
                        mock_fit.assert_called_once()

    @pytest.mark.parametrize("model_name", ["NHP", "RMTPP"])
    def test_runner_test_step(self, temporary_directory, model_name):
        """Test testing step through runner."""
        config_dict = self.test_configs[model_name]
        config_path = self._create_temp_config_file(config_dict, temporary_directory)

        experiment_id = f"{model_name}_test"

        # Mock data loaders
        mock_test_loader = Mock()

        with patch("easy_tpp.preprocess.data_loader.TPPDataModule.setup"):
            with patch(
                "easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader",
                return_value=mock_test_loader,
            ):
                with patch(
                    "pytorch_lightning.Trainer.test", return_value=[{"test_loss": 0.5}]
                ) as mock_test:
                    with patch("builtins.open", create=True) as mock_open:
                        # Build config and create trainer
                        config = RunnerConfig.load_from_yaml_file(config_path)
                        config.experiment_id = experiment_id
                        config.dataset_id = "test_data"
                        trainer = Trainer(config, output_dir=str(temporary_directory))

                        # Test testing
                        results = trainer.test()

                        # Verify testing was called
                        mock_test.assert_called_once()
                        assert results is not None

    @pytest.mark.parametrize("model_name", ["NHP", "RMTPP"])
    def test_runner_predict_step(self, temporary_directory, model_name):
        """Test prediction step through runner."""
        config_dict = self.test_configs[model_name]
        config_path = self._create_temp_config_file(config_dict, temporary_directory)

        experiment_id = f"{model_name}_test"

        # Mock data loaders and prediction components
        mock_test_loader = Mock()

        with patch("easy_tpp.preprocess.data_loader.TPPDataModule.setup"):
            with patch(
                "easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader",
                return_value=mock_test_loader,
            ):
                with patch(
                    "pytorch_lightning.Trainer.predict", return_value=[Mock()]
                ) as mock_predict:
                    with patch.object(Trainer, "model") as mock_model:
                        mock_model.format_and_save_simulations = Mock()
                        mock_model.simulations = Mock()

                        with patch(
                            "easy_tpp.evaluate.distribution_analysis_helper.NTPPComparator"
                        ):
                            # Build config and create trainer
                            config = RunnerConfig.load_from_yaml_file(config_path)
                            config.experiment_id = experiment_id
                            config.dataset_id = "test_data"
                            trainer = Trainer(
                                config, output_dir=str(temporary_directory)
                            )
                            trainer.model = mock_model

                            # Test prediction
                            predictions = trainer.predict()

                            # Verify prediction was called
                            mock_predict.assert_called_once()
                            assert predictions is not None

    def test_hawkes_model_runner(self, temporary_directory):
        """Test Hawkes model specifically through runner."""
        config_dict = self.test_configs["Hawkes"]
        config_path = self._create_temp_config_file(
            config_dict, temporary_directory, experiment_id="Hawkes_test"
        )

        experiment_id = "Hawkes_test"

        with patch("easy_tpp.preprocess.data_loader.TPPDataModule.setup"):
            with patch("easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader"):
                # Build config and create trainer
                config = RunnerConfig.load_from_yaml_file(config_path)
                config.experiment_id = experiment_id
                config.dataset_id = "test_data"
                trainer = Trainer(config, output_dir=str(temporary_directory))

                # Verify Hawkes model was created
                assert trainer is not None
                assert trainer.model is not None
                assert trainer.model_id == "HawkesModel"

    def test_config_override_through_runner(self, temporary_directory):
        """Test configuration override capabilities through runner."""
        config_dict = self.test_configs["NHP"]
        experiment_id = "NHP_test"
        config_path = self._create_temp_config_file(
            config_dict, temporary_directory, experiment_id=experiment_id
        )

        with patch("easy_tpp.preprocess.data_loader.TPPDataModule.setup"):
            with patch(
                "easy_tpp.preprocess.data_loader.TPPDataModule.train_dataloader"
            ):
                with patch(
                    "easy_tpp.preprocess.data_loader.TPPDataModule.val_dataloader"
                ):
                    # Build config with custom output directory
                    config = RunnerConfig.load_from_yaml_file(config_path)
                    config.experiment_id = experiment_id
                    config.dataset_id = "test_data"
                    custom_output_dir = str(temporary_directory / "custom_output")
                    trainer = Trainer(config, output_dir=custom_output_dir)

                    # Verify output directory was overridden
                    assert trainer.dirpath == custom_output_dir

    def test_checkpoint_loading_through_runner(self, temporary_directory):
        """Test checkpoint loading capabilities through runner."""
        config_dict = self.test_configs["NHP"]
        config_path = self._create_temp_config_file(
            config_dict, temporary_directory, experiment_id=experiment_id
        )

        experiment_id = "NHP_test"

        # Create a dummy checkpoint file
        checkpoint_dir = temporary_directory / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        dummy_checkpoint = checkpoint_dir / "best.ckpt"
        dummy_checkpoint.touch()

        with patch("easy_tpp.preprocess.data_loader.TPPDataModule.setup"):
            # Build config and create trainer with checkpoint
            config = RunnerConfig.load_from_yaml_file(config_path)
            config.experiment_id = experiment_id
            config.dataset_id = "test_data"
            trainer = Trainer(
                config, checkpoint_path="best", output_dir=str(checkpoint_dir)
            )

            # Verify checkpoint path is correctly set
            assert trainer.checkpoint_path_ is not None
            assert "best.ckpt" in trainer.checkpoint_path_

    def test_multiple_models_comparison_through_runner(self, temporary_directory):
        """Test creating and comparing multiple models through runner."""
        trainers = {}

        for model_name in ["NHP", "RMTPP"]:
            config_dict = self.test_configs[model_name]
            config_path = self._create_temp_config_file(
                config_dict, temporary_directory
            )

            experiment_id = f"{model_name}_test"

            with patch("easy_tpp.preprocess.data_loader.TPPDataModule.setup"):
                config = RunnerConfig.load_from_yaml_file(config_path)
                config.experiment_id = experiment_id
                config.dataset_id = "test_data"
                trainer = Trainer(
                    config, output_dir=str(temporary_directory / model_name)
                )
                trainers[model_name] = trainer
        # Verify all models were created successfully
        assert len(trainers) == 2
        assert all(t.model is not None for t in trainers.values())
        assert trainers["NHP"].model_id == "NHP"
        assert trainers["RMTPP"].model_id == "RMTPP"
        # Verify models have different configurations
        nhp_hidden_size = trainers["NHP"].model.hidden_size
        rmtpp_hidden_size = trainers["RMTPP"].model.hidden_size
        assert (
            nhp_hidden_size != rmtpp_hidden_size
        )  # They should be different based on our config
