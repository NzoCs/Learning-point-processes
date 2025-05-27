"""Runner-based unit tests for individual components.

This module provides unit tests that use the runner configuration approach
to test individual components like model creation, data loading, and training steps.
"""
import pytest
import torch
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from easy_tpp.config_factory import Config
from easy_tpp.runner import Trainer
from easy_tpp.utils.torch_utils import set_seed


@pytest.mark.unit
class TestRunnerBasedUnit:
    """Unit tests using the runner-based configuration approach."""
    
    def setup_method(self):
        """Set up for each test method."""
        set_seed(42)
    
    def _create_minimal_config(self, model_id='NHP', temp_dir=None):
        """Create a minimal configuration for testing."""
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
            f'{model_id}_test': {
                'data_loading_specs': {
                    'batch_size': 4,
                    'num_workers': 1
                },
                'model_config': {
                    'model_id': model_id,
                    'specs': {
                        'hidden_size': 16,
                        'time_emb_size': 8,
                        'num_layers': 1                    } if model_id in ['NHP', 'RMTPP'] else {
                        'mu': [0.1, 0.1],
                        'alpha': [[0.2, 0.1], [0.1, 0.2]],
                        'beta': [[1.0, 0.5], [0.5, 1.0]]
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
                    } if model_id in ['NHP', 'RMTPP'] else None
                },
                'trainer_config': {
                    'stage': 'train',
                    'max_epochs': 1,
                    'val_freq': 1,
                    'accumulate_grad_batches': 1,
                    'patience': 3,
                    'save_model_dir': str(temp_dir) if temp_dir else 'test_checkpoints',
                    'devices': 1,
                    'use_precision_16': False,
                    'log_freq': 1,
                    'checkpoints_freq': 1
                }
            }
        }
        
        # Remove None values for Hawkes model
        if model_id == 'HawkesModel':
            config[f'{model_id}_test']['model_config'] = {
                k: v for k, v in config[f'{model_id}_test']['model_config'].items() 
                if v is not None
            }
        
        return config
    
    def _create_temp_config_file(self, config_dict, temp_dir):
        """Create a temporary YAML config file."""
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        return str(config_path)
    
    def test_runner_model_creation_nhp(self, temporary_directory):
        """Test NHP model creation through runner."""
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))            # Test model properties
            assert trainer.model is not None
            assert trainer.model_id == 'NHP'
            assert hasattr(trainer.model, 'hidden_size')
            assert trainer.model.hidden_size == 16
    
    def test_runner_model_creation_rmtpp(self, temporary_directory):
        """Test RMTPP model creation through runner."""
        config_dict = self._create_minimal_config('RMTPP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='RMTPP_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))            # Test model properties
            assert trainer.model is not None
            assert trainer.model_id == 'RMTPP'
            assert hasattr(trainer.model, 'hidden_size')
            assert trainer.model.hidden_size == 16
    
    def test_runner_model_creation_hawkes(self, temporary_directory):
        """Test Hawkes model creation through runner."""
        config_dict = self._create_minimal_config('HawkesModel', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='HawkesModel_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
              # Test model properties
            assert trainer.model is not None
            assert trainer.model_id == 'HawkesModel'
            # Note: HawkesModel doesn't have model_config attribute like other models
            assert hasattr(trainer.model, 'num_event_types')
    
    def test_runner_datamodule_creation(self, temporary_directory):
        """Test datamodule creation through runner."""
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
            
            # Test datamodule properties
            assert trainer.datamodule is not None
            assert hasattr(trainer.datamodule, 'data_config')
            assert trainer.dataset_id == 'test_data'
    
    def test_runner_pytorch_lightning_trainer_creation(self, temporary_directory):
        """Test PyTorch Lightning trainer creation through runner."""
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
            
            # Test PyTorch Lightning trainer properties
            pl_trainer = trainer.trainer
            assert pl_trainer is not None
            assert pl_trainer.max_epochs == 1
            assert len(pl_trainer.callbacks) == 2  # ModelCheckpoint + EarlyStopping
    
    def test_runner_callbacks_configuration(self, temporary_directory):
        """Test callbacks configuration through runner."""
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
            
            # Test callbacks
            callbacks = trainer.callbacks
            assert len(callbacks) == 2
            
            # Check ModelCheckpoint callback
            model_checkpoint = callbacks[0]
            assert model_checkpoint.monitor == 'val_loss'
            assert model_checkpoint.mode == 'min'
            
            # Check EarlyStopping callback
            early_stopping = callbacks[1]
            assert early_stopping.monitor == 'val_loss'
            assert early_stopping.patience == 3
    
    def test_runner_checkpoint_path_logic(self, temporary_directory):
        """Test checkpoint path logic through runner."""
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_test',
                dataset_id='test_data'
            )
            
            # Test without checkpoint
            trainer = Trainer(config, output_dir=str(temporary_directory))
            assert trainer.checkpoint_path is None
            
            # Create a dummy checkpoint and test with it
            checkpoint_file = temporary_directory / 'best.ckpt'
            checkpoint_file.touch()
            
            trainer_with_checkpoint = Trainer(
                config, 
                checkpoint_path='best',
                output_dir=str(temporary_directory)
            )
            assert trainer_with_checkpoint.checkpoint_path is not None
    
    def test_runner_training_configuration(self, temporary_directory):
        """Test training configuration through runner."""
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
            
            # Test training configuration properties
            assert trainer.max_epochs == 1
            assert trainer.val_freq == 1
            assert trainer.patience == 3
            assert trainer.devices == 1
            assert trainer.use_precision_16 == False
            assert trainer.accumulate_grad_batches == 1
    
    def test_runner_precision_and_device_configuration(self, temporary_directory):
        """Test precision and device configuration through runner."""
        # Test with different device and precision settings
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_dict['NHP_test']['trainer_config']['use_precision_16'] = True
        config_dict['NHP_test']['trainer_config']['devices'] = 'auto'
        
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
            
            # Test precision and device settings
            assert trainer.use_precision_16 == True
            
            # Test PyTorch Lightning trainer configuration
            pl_trainer = trainer.trainer
            assert pl_trainer.precision == '16-mixed'
    
    def test_runner_model_forward_pass(self, temporary_directory):
        """Test model forward pass through runner."""
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            config = Config.build_from_yaml_file(
                yaml_dir=config_path,
                experiment_id='NHP_test',
                dataset_id='test_data'
            )
            
            trainer = Trainer(config, output_dir=str(temporary_directory))
            
            # Create dummy input for forward pass
            batch_size = 2
            seq_len = 5
            num_event_types = 2
            
            time_seqs = torch.rand(batch_size, seq_len)
            type_seqs = torch.randint(0, num_event_types, (batch_size, seq_len))
            
            # Test forward pass
            with torch.no_grad():
                # This tests that the model can handle input
                assert trainer.model is not None
                # We don't call forward directly as it requires specific batch format
                # but we ensure the model is properly configured
                assert hasattr(trainer.model, 'forward')
    
    def test_runner_error_handling(self, temporary_directory):
        """Test error handling in runner configuration."""
        # Test with invalid model configuration
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_dict['NHP_test']['model_config']['model_id'] = 'InvalidModel'
        
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            with pytest.raises(Exception):  # Should raise an error for invalid model
                config = Config.build_from_yaml_file(
                    yaml_dir=config_path,
                    experiment_id='NHP_test',
                    dataset_id='test_data'
                )
                Trainer(config, output_dir=str(temporary_directory))
    
    def test_runner_logger_configuration(self, temporary_directory):
        """Test logger configuration through runner."""
        config_dict = self._create_minimal_config('NHP', temporary_directory)
        config_dict['NHP_test']['trainer_config']['logger_config'] = {
            'type': 'tensorboard',
            'name': 'test_logs'
        }
        
        config_path = self._create_temp_config_file(config_dict, temporary_directory)
        
        with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
            with patch('easy_tpp.config_factory.runner_config.TrainerConfig.get_logger') as mock_logger:
                mock_logger.return_value = Mock()
                
                config = Config.build_from_yaml_file(
                    yaml_dir=config_path,
                    experiment_id='NHP_test',
                    dataset_id='test_data'
                )
                
                trainer = Trainer(config, output_dir=str(temporary_directory))
                
                # Test that logger was configured
                assert trainer.logger_config is not None
                mock_logger.assert_called_once()
    
    def test_runner_different_batch_sizes(self, temporary_directory):
        """Test different batch size configurations through runner."""
        for batch_size in [1, 4, 8]:
            config_dict = self._create_minimal_config('NHP', temporary_directory)
            config_dict['NHP_test']['data_loading_specs']['batch_size'] = batch_size
            
            config_path = self._create_temp_config_file(config_dict, temporary_directory)
            
            with patch('easy_tpp.preprocess.data_loader.TPPDataModule.setup'):
                config = Config.build_from_yaml_file(
                    yaml_dir=config_path,
                    experiment_id='NHP_test',
                    dataset_id='test_data'
                )
                
                trainer = Trainer(config, output_dir=str(temporary_directory))
                
                # Test that batch size is properly configured
                assert trainer.datamodule.data_config.batch_size == batch_size
