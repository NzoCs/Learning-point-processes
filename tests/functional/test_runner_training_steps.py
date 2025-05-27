"""Functional tests for training, testing, and validation steps using runner approach.

This module provides functional tests that specifically test the training,
testing, and validation steps using the runner-based configuration approach.
"""
import pytest
import torch
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from easy_tpp.config_factory import Config
from easy_tpp.runner import Trainer
from easy_tpp.utils.torch_utils import set_seed


@pytest.mark.functional
class TestRunnerTrainingSteps:
    """Functional tests for training, testing, and validation steps through runner."""
    
    def setup_method(self):
        """Set up for each test method."""
        set_seed(42)
    
    def _create_training_config(self, model_id='NHP', temp_dir=None):
        """Create a training configuration."""
        config = {
            'pipeline_config_id': 'runner_config',
            'data': {
                'test_data': {
                    'data_format': 'json',
                    'train_dir': 'test/data',
                    'valid_dir': 'test/data',
                    'test_dir': 'test/data',
                    'data_specs': {
                        'num_event_types': 2,
                        'pad_token_id': 2,
                        'padding_side': 'left'
                    }
                }
            },
            f'{model_id}_train': {
                'data_loading_specs': {
                    'batch_size': 4,
                    'num_workers': 1
                },
                'model_config': {
                    'model_id': model_id,
                    'specs': {
                        'hidden_size': 16,
                        'time_emb_size': 8,
                        'num_layers': 1
                    } if model_id in ['NHP', 'RMTPP'] else {
                        'mu': [0.1, 0.1],
                        'alpha': [[0.2, 0], [0, 0.4]],
                        'beta': [[1.0, 0], [0, 2.0]]
                    },
                    'thinning': {
                        'num_exp': 10,
                        'over_sample_rate': 1.5,
                        'dtime_max': 5,
                        'num_sample': 5
                    },
                    'base_config': {
                        'lr': 0.01,
                        'lr_scheduler': False
                    } if model_id in ['NHP', 'RMTPP'] else None,
                    'simulation_config': {
                        'start_time': 10,
                        'end_time': 20,
                        'max_sim_events': 50
                    } if model_id in ['NHP', 'RMTPP'] else None
                },
                'trainer_config': {
                    'stage': 'train',
                    'max_epochs': 2,
                    'val_freq': 1,
                    'accumulate_grad_batches': 1,
                    'patience': 5,
                    'save_model_dir': str(temp_dir) if temp_dir else 'test_checkpoints',
                    'devices': 1,
                    'use_precision_16': False,
                    'log_freq': 1,
                    'checkpoints_freq': 1,
                    'logger_config': {
                        'type': 'tensorboard',
                        'name': 'test_logs'
                    }
                }
            }
        }
        
        # Clean up None values for specific models
        if model_id == 'HawkesModel':
            config[f'{model_id}_train']['model_config'] = {
                k: v for k, v in config[f'{model_id}_train']['model_config'].items() 
                if v is not None
            }
        
        return config
    
    def _create_temp_config_file(self, config_dict, temp_dir):
        """Create a temporary YAML config file."""
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        return str(config_path)
    
    @pytest.mark.parametrize("model_id", ["NHP", "RMTPP"])
    def test_training_step_execution(self, temporary_directory, model_id):
        """Test training step execution through runner."""
        config_dict = self._create_training_config(model_id, temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        # Mock data loaders
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup') as mock_setup:
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.train_dataloader', return_value=mock_train_loader) as mock_train_dl:
                with patch('easy_tpp.preprocess.data_loader.TPPDataModule.val_dataloader', return_value=mock_val_loader) as mock_val_dl:
                    with patch('pytorch_lightning.Trainer.fit') as mock_fit:
                        # Build config and create trainer
                        config = Config.build_from_yaml_file(
                            yaml_dir=config_path,
                            experiment_id=f'{model_id}_train',
                            dataset_id='test_data'
                        )
                        
                        trainer = Trainer(config, output_dir=str(temporary_directory))
                        
                        # Execute training step
                        trainer.train()
                        
                        # Verify all components were called correctly
                        mock_setup.assert_called_with(stage='fit')
                        mock_train_dl.assert_called_once()
                        mock_val_dl.assert_called_once()
                        mock_fit.assert_called_once()
                        
                        # Verify fit was called with correct parameters
                        fit_call = mock_fit.call_args
                        assert fit_call.kwargs['model'] == trainer.model
                        assert fit_call.kwargs['train_dataloaders'] == mock_train_loader
                        assert fit_call.kwargs['val_dataloaders'] == mock_val_loader
    
    @pytest.mark.parametrize("model_id", ["NHP", "RMTPP"])
    def test_testing_step_execution(self, temporary_directory, model_id):
        """Test testing step execution through runner."""
        config_dict = self._create_training_config(model_id, temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        # Mock test data loader and results
        mock_test_loader = MagicMock()
        mock_test_results = [{'test_loss': 0.42, 'test_accuracy': 0.85}]
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup') as mock_setup:
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader', return_value=mock_test_loader) as mock_test_dl:
                with patch('pytorch_lightning.Trainer.test', return_value=mock_test_results) as mock_test:
                    with patch('builtins.open', create=True) as mock_open:
                        with patch('json.dump') as mock_json_dump:
                            # Build config and create trainer
                            config = Config.build_from_yaml_file(
                                yaml_dir=config_path,
                                experiment_id=f'{model_id}_train',
                                dataset_id='test_data'
                            )
                            
                            trainer = Trainer(config, output_dir=str(temporary_directory))
                            
                            # Execute testing step
                            results = trainer.test()
                            
                            # Verify all components were called correctly
                            mock_setup.assert_called_with(stage='test')
                            mock_test_dl.assert_called_once()
                            mock_test.assert_called_once()
                            
                            # Verify test was called with correct parameters
                            test_call = mock_test.call_args
                            assert test_call.kwargs['model'] == trainer.model
                            assert test_call.kwargs['dataloaders'] == mock_test_loader
                            
                            # Verify results were saved
                            assert results == mock_test_results
                            mock_json_dump.assert_called_once()
    
    @pytest.mark.parametrize("model_id", ["NHP", "RMTPP"])
    def test_prediction_step_execution(self, temporary_directory, model_id):
        """Test prediction step execution through runner."""
        config_dict = self._create_training_config(model_id, temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        # Mock prediction components
        mock_test_loader = MagicMock()
        mock_predictions = [MagicMock(), MagicMock()]
        mock_simulations = MagicMock()
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup') as mock_setup:
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader', return_value=mock_test_loader) as mock_test_dl:
                with patch('pytorch_lightning.Trainer.predict', return_value=mock_predictions) as mock_predict:
                    with patch('easy_tpp.evaluate.new_comparator.NewDistribComparator') as mock_comparator:
                        # Build config and create trainer
                        config = Config.build_from_yaml_file(
                            yaml_dir=config_path,
                            experiment_id=f'{model_id}_train',
                            dataset_id='test_data'
                        )
                        
                        trainer = Trainer(config, output_dir=str(temporary_directory))
                        
                        # Mock model methods
                        trainer.model.format_and_save_simulations = MagicMock()
                        trainer.model.simulations = mock_simulations
                        
                        # Execute prediction step
                        predictions = trainer.predict()
                        
                        # Verify all components were called correctly
                        mock_setup.assert_called_with(stage='predict')
                        mock_test_dl.assert_called_once()
                        mock_predict.assert_called_once()
                        
                        # Verify predict was called with correct parameters
                        predict_call = mock_predict.call_args
                        assert predict_call.kwargs['model'] == trainer.model
                        assert predict_call.kwargs['dataloaders'] == mock_test_loader
                        
                        # Verify simulation and comparison methods were called
                        trainer.model.format_and_save_simulations.assert_called_once()
                        mock_comparator.assert_called_once()
                        
                        # Verify results
                        assert predictions == mock_predictions
    
    def test_validation_step_during_training(self, temporary_directory):
        """Test validation step during training through runner."""
        config_dict = self._create_training_config('NHP', temporary_directory)
        # Set validation frequency
        config_dict['NHP_train']['trainer_config']['val_freq'] = 1
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.train_dataloader'):
                with patch('easy_tpp.preprocess.data_loader.TPPDataModule.val_dataloader'):
                    # Build config and create trainer
                    config = Config.build_from_yaml_file(
                        yaml_dir=config_path,
                        experiment_id='NHP_train',
                        dataset_id='test_data'
                    )
                    
                    trainer = Trainer(config, output_dir=str(temporary_directory))
                    
                    # Verify validation frequency is set correctly
                    pl_trainer = trainer.trainer
                    assert pl_trainer.check_val_every_n_epoch == 1
    
    def test_checkpoint_saving_during_training(self, temporary_directory):
        """Test checkpoint saving during training through runner."""
        config_dict = self._create_training_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            # Build config and create trainer
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_train',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
            
            # Verify checkpoint callback is configured
            callbacks = trainer.callbacks
            model_checkpoint = callbacks[0]  # First callback should be ModelCheckpoint
            
            assert model_checkpoint.dirpath == str(temporary_directory)
            assert model_checkpoint.filename == "best"
            assert model_checkpoint.monitor == 'val_loss'
            assert model_checkpoint.save_top_k == 1
            assert model_checkpoint.mode == 'min'
    
    def test_early_stopping_during_training(self, temporary_directory):
        """Test early stopping during training through runner."""
        config_dict = self._create_training_config('NHP', temporary_directory)
        # Set patience for early stopping
        config_dict['NHP_train']['trainer_config']['patience'] = 3
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            # Build config and create trainer
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_train',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
            
            # Verify early stopping callback is configured
            callbacks = trainer.callbacks
            early_stopping = callbacks[1]  # Second callback should be EarlyStopping
            
            assert early_stopping.monitor == 'val_loss'
            assert early_stopping.patience == 3
            assert early_stopping.mode == 'min'
    
    def test_training_with_checkpoint_resumption(self, temporary_directory):
        """Test training with checkpoint resumption through runner."""
        config_dict = self._create_training_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        # Create a dummy checkpoint file
        checkpoint_file = temporary_directory / 'best.ckpt'
        checkpoint_file.touch()
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.train_dataloader'):
                with patch('easy_tpp.preprocess.data_loader.TPPDataModule.val_dataloader'):
                    with patch('pytorch_lightning.Trainer.fit') as mock_fit:
                        # Build config and create trainer with checkpoint
                        config = Config.build_from_yaml_file(
                            yaml_dir=config_path,
                            experiment_id='NHP_train',
                            dataset_id='test_data'
                        )
                        
                        trainer = Trainer(
                            config, 
                            checkpoint_path='best',
                            output_dir=str(temporary_directory)
                        )
                        
                        # Execute training
                        trainer.train()
                        
                        # Verify fit was called with checkpoint path
                        fit_call = mock_fit.call_args
                        assert 'best.ckpt' in str(fit_call.kwargs['ckpt_path'])
    
    def test_testing_with_checkpoint_loading(self, temporary_directory):
        """Test testing with checkpoint loading through runner."""
        config_dict = self._create_training_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        # Create a dummy checkpoint file
        checkpoint_file = temporary_directory / 'best.ckpt'
        checkpoint_file.touch()
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader'):
                with patch('pytorch_lightning.Trainer.test', return_value=[{'test_loss': 0.5}]) as mock_test:
                    with patch('builtins.open', create=True):
                        with patch('json.dump'):
                            # Build config and create trainer with checkpoint
                            config = Config.build_from_yaml_file(
                                yaml_dir=config_path,
                                experiment_id='NHP_train',
                                dataset_id='test_data'
                            )
                            
                            trainer = Trainer(
                                config, 
                                checkpoint_path='best',
                                output_dir=str(temporary_directory)
                            )
                            
                            # Execute testing
                            trainer.test()
                            
                            # Verify test was called with checkpoint path
                            test_call = mock_test.call_args
                            assert 'best.ckpt' in str(test_call.kwargs['ckpt_path'])
    
    def test_prediction_with_checkpoint_loading(self, temporary_directory):
        """Test prediction with checkpoint loading through runner."""
        config_dict = self._create_training_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        # Create a dummy checkpoint file
        checkpoint_file = temporary_directory / 'best.ckpt'
        checkpoint_file.touch()
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader'):
                with patch('pytorch_lightning.Trainer.predict', return_value=[Mock()]) as mock_predict:
                    with patch('easy_tpp.evaluate.new_comparator.NewDistribComparator'):
                        # Build config and create trainer with checkpoint
                        config = Config.build_from_yaml_file(
                            yaml_dir=config_path,
                            experiment_id='NHP_train',
                            dataset_id='test_data'
                        )
                        
                        trainer = Trainer(
                            config, 
                            checkpoint_path='best',
                            output_dir=str(temporary_directory)
                        )
                        
                        # Mock model methods
                        trainer.model.format_and_save_simulations = MagicMock()
                        trainer.model.simulations = MagicMock()
                        
                        # Execute prediction
                        trainer.predict()
                        
                        # Verify predict was called with checkpoint path
                        predict_call = mock_predict.call_args
                        assert 'best.ckpt' in str(predict_call.kwargs['ckpt_path'])
    
    def test_training_step_error_handling(self, temporary_directory):
        """Test error handling during training step through runner."""
        config_dict = self._create_training_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.train_dataloader'):
                with patch('easy_tpp.preprocess.data_loader.TPPDataModule.val_dataloader'):
                    with patch('pytorch_lightning.Trainer.fit', side_effect=Exception("Training failed")):
                        # Build config and create trainer
                        config = Config.build_from_yaml_file(
                            yaml_dir=config_path,
                            experiment_id='NHP_train',
                            dataset_id='test_data'
                        )
                        
                        trainer = Trainer(config, output_dir=str(temporary_directory))
                        
                        # Verify that training error is properly raised
                        with pytest.raises(Exception, match="Training failed"):
                            trainer.train()
    
    def test_different_optimizer_configurations(self, temporary_directory):
        """Test different optimizer configurations through runner."""
        for lr in [0.001, 0.01, 0.1]:
            config_dict = self._create_training_config('NHP', temporary_directory)
            config_dict['NHP_train']['model_config']['base_config']['lr'] = lr
            config_path = self._create_temp_config_file(config_dict, temporary_directory)
            
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
                # Build config and create trainer
                config = Config.build_from_yaml_file(
                    yaml_dir=config_path,
                    experiment_id='NHP_train',
                    dataset_id='test_data'
                )
                
                trainer = Trainer(config, output_dir=str(temporary_directory))
                
                # Verify learning rate is properly set
                assert trainer.model.model_config.lr == lr
    
    def test_complete_training_pipeline(self, temporary_directory):
        """Test complete training pipeline through runner."""
        config_dict = self._create_training_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        # Mock all components for complete pipeline
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.train_dataloader'):
                with patch('easy_tpp.preprocess.data_loader.TPPDataModule.val_dataloader'):
                    with patch('easy_tpp.preprocess.data_loader.TPPDataModule.test_dataloader'):
                        with patch('pytorch_lightning.Trainer.fit') as mock_fit:
                            with patch('pytorch_lightning.Trainer.test', return_value=[{'test_loss': 0.5}]) as mock_test:
                                with patch('pytorch_lightning.Trainer.predict', return_value=[Mock()]) as mock_predict:
                                    with patch('builtins.open', create=True):
                                        with patch('json.dump'):
                                            with patch('easy_tpp.evaluate.new_comparator.NewDistribComparator'):
                                                # Build config and create trainer
                                                config = Config.build_from_yaml_file(
                                                    yaml_dir=config_path,
                                                    experiment_id='NHP_train',
                                                    dataset_id='test_data'
                                                )
                                                
                                                trainer = Trainer(config, output_dir=str(temporary_directory))
                                                trainer.model.format_and_save_simulations = MagicMock()
                                                trainer.model.simulations = MagicMock()
                                                
                                                # Execute complete pipeline
                                                trainer.train()
                                                trainer.test()
                                                trainer.predict()
                                                
                                                # Verify all steps were executed
                                                mock_fit.assert_called_once()
                                                mock_test.assert_called_once()
                                                mock_predict.assert_called_once()
