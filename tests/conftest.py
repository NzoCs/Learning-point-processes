# Main test configuration and fixtures
import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytorch_lightning as pl
from omegaconf import OmegaConf

# Import easy_tpp modules
from easy_tpp.config_factory import ModelConfig, DataConfig, RunnerConfig
from easy_tpp.utils.torch_utils import set_device, set_seed


@pytest.fixture(scope="session")
def device():
    """Fixture to provide device for testing."""
    return set_device(-1)  # Use CPU by default for tests


@pytest.fixture(scope="session")
def gpu_device():
    """Fixture to provide GPU device if available."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    pytest.skip("GPU not available")


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset random seed before each test for reproducibility."""
    set_seed(42)


@pytest.fixture
def sample_model_config():
    """Create a sample model configuration for testing."""
    config_dict = {
        'model_id': 'NHP',
        'hidden_size': 32,
        'num_event_types': 5,
        'num_event_types_pad': 6,
        'max_seq_len': 100,
        'lr': 0.001,
        'batch_size': 16,
        'device_id': -1,  # CPU by default
        'thinning': {
            'n_samples': 1000,
            'patience_counter': 200
        }
    }
    return ModelConfig(**config_dict)


@pytest.fixture
def sample_data_config():
    """Create a sample data configuration for testing."""
    config_dict = {
        'dataset_name': 'test_dataset',
        'data_format': 'pkl',
        'train_dir': 'test_data/train',
        'valid_dir': 'test_data/valid', 
        'test_dir': 'test_data/test',
        'num_event_types': 5,
        'pad_token_id': 0,
        'padding_side': 'left',
        'truncation_side': 'right',
        'max_seq_len': 100
    }
    return DataConfig(**config_dict)


@pytest.fixture
def sample_runner_config():
    """Create a sample runner configuration for testing."""
    config_dict = {
        'base_dir': './test_output',
        'logger_config': {
            'logger_type': 'none'
        },
        'trainer_config': {
            'max_epochs': 1,
            'enable_checkpointing': False,
            'logger': False,
            'enable_progress_bar': False,
            'enable_model_summary': False
        }
    }
    return RunnerConfig(**config_dict)


@pytest.fixture
def sample_batch_data():
    """Create sample batch data for model testing."""
    batch_size = 4
    max_seq_len = 10
    num_event_types = 5
    
    # Time intervals (dt)
    time_seqs = torch.rand(batch_size, max_seq_len) * 2.0
    
    # Event types
    type_seqs = torch.randint(1, num_event_types + 1, (batch_size, max_seq_len))
    
    # Sequence lengths
    seq_lens = torch.randint(5, max_seq_len + 1, (batch_size,))
    
    # Attention mask
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(seq_lens):
        attention_mask[i, :length] = True
    
    return {
        'time_seqs': time_seqs,
        'type_seqs': type_seqs,
        'seq_lens': seq_lens,
        'attention_mask': attention_mask,
        'batch_non_pad_mask': attention_mask,
        'type_mask': attention_mask
    }


@pytest.fixture
def temporary_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_trainer():
    """Create a mock PyTorch Lightning trainer."""
    trainer = Mock(spec=pl.Trainer)
    trainer.fit = Mock()
    trainer.test = Mock()
    trainer.validate = Mock()
    trainer.predict = Mock()
    return trainer


@pytest.fixture
def sample_event_sequences():
    """Create sample event sequences for data processing tests."""
    sequences = [
        {
            'time_since_start': [0.1, 0.3, 0.7, 1.2, 1.8],
            'time_since_last_event': [0.1, 0.2, 0.4, 0.5, 0.6],
            'type_event': [1, 2, 1, 3, 2],
            'idx_event': 0
        },
        {
            'time_since_start': [0.2, 0.5, 0.9],
            'time_since_last_event': [0.2, 0.3, 0.4],
            'type_event': [2, 1, 3],
            'idx_event': 1
        }
    ]
    return sequences


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, batch_data, num_batches=3):
        self.batch_data = batch_data
        self.num_batches = num_batches
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        self.current_batch += 1
        return self.batch_data
    
    def __len__(self):
        return self.num_batches


@pytest.fixture
def mock_dataloader(sample_batch_data):
    """Create a mock data loader."""
    return MockDataLoader(sample_batch_data)


# Device testing utilities
def check_device_consistency(model, expected_device):
    """Check if all model parameters are on the expected device."""
    for param in model.parameters():
        assert param.device == expected_device, f"Parameter on {param.device}, expected {expected_device}"


def check_tensor_device(tensor, expected_device):
    """Check if tensor is on expected device."""
    assert tensor.device == expected_device, f"Tensor on {tensor.device}, expected {expected_device}"


# Skip decorators for specific conditions
skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), 
    reason="GPU not available"
)

skip_if_no_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


# Custom markers
pytestmark = pytest.mark.unit
