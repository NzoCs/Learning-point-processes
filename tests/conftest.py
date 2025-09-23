# Main test configuration and fixtures
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

# Import easy_tpp modules
from easy_tpp.configs import DataConfig, ModelConfig, RunnerConfig
from easy_tpp.utils.torch_utils import set_device, set_seed

DEFAULT_NUM_EVENT_TYPES = 5
DEFAULT_MAX_SEQ_LEN = 100
DEFAULT_DATASET_NAME = "test_dataset"
DEFAULT_MODEL_ID = "NHP"
DEFAULT_COMPUTE_SIMULATION = True


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
    num_event_types = DEFAULT_NUM_EVENT_TYPES
    model_id = DEFAULT_MODEL_ID
    if model_id in ["NHP", "RMTPP"]:
        specs = {"hidden_size": 32, "max_seq_len": DEFAULT_MAX_SEQ_LEN}
    elif model_id == "Hawkes":
        specs = {
            "mu": [0.5] * num_event_types,
            "alpha": [[0.8] * num_event_types for _ in range(num_event_types)],
            "beta": [[1.2] * num_event_types for _ in range(num_event_types)],
        }
    else:
        specs = {}
    config_dict = {
        "model_id": model_id,
        "num_event_types": num_event_types,
        "num_event_types_pad": num_event_types + 1,
        "compute_simulation": DEFAULT_COMPUTE_SIMULATION,
        "thinning": {
            "num_sample": 10,
            "num_exp": 200,
            "num_steps": 10,
            "over_sample_rate": 1.5,
            "num_samples_boundary": 5,
            "dtime_max": 5.0,
        },
        "simulation_config": {
            "start_time": 0.0,
            "end_time": 10.0,
            "batch_size": 16,
            "max_sim_events": 1000,
            "seed": 42,
        },
        "specs": specs,
    }
    return ModelConfig.from_dict(config_dict)


@pytest.fixture
def sample_data_config():
    """Create a sample data configuration for testing."""
    config_dict = {
        "dataset_name": DEFAULT_DATASET_NAME,
        "data_format": "pkl",
        "train_dir": "test_data/train",
        "valid_dir": "test_data/valid",
        "test_dir": "test_data/test",
        "num_event_types": DEFAULT_NUM_EVENT_TYPES,
        "pad_token_id": 0,
        "padding_side": "left",
        "truncation_side": "right",
        "max_seq_len": DEFAULT_MAX_SEQ_LEN,
    }
    return DataConfig(**config_dict)


@pytest.fixture
def sample_runner_config():
    """Create a sample runner configuration for testing."""
    from easy_tpp.configs.data_config import DataConfig
    from easy_tpp.configs.model_config import ModelConfig
    from easy_tpp.configs.runner_config import TrainerConfig

    # Create trainer config
    trainer_config = TrainerConfig(
        max_epochs=1,
        dataset_id="test_dataset",
        model_id="NHP",  # Use a valid model
        batch_size=32,
        logger_config={"logger_type": "none"},
    )

    # Create a minimal data config
    data_config = DataConfig.parse_from_yaml_config(
        {
            "dataset_id": "test_dataset",
            "data_specs": {"num_event_types": 5, "pad_token_id": 0},
            "data_loading_specs": {},
        }
    )
    # Create a minimal model config
    model_config = ModelConfig.parse_from_yaml_config(
        {
            "model_id": "NHP",  # Use a valid model
            "base_config": {"lr": 0.001, "max_epochs": 1},
            "specs": {
                "hidden_size": 32,
                "time_emb_size": 16,
                "use_ln": True,
                "num_layers": 2,
                "num_heads": 4,
            },
            "num_event_types": 5,
        }
    )

    return RunnerConfig(
        trainer_config=trainer_config,
        model_config=model_config,
        data_config=data_config,
    )


@pytest.fixture
def sample_batch_data():
    """Create sample batch data for model testing."""
    batch_size = 4
    max_seq_len = 10
    num_event_types = 5

    # Time intervals (dt)
    time_seqs = torch.rand(batch_size, max_seq_len) * 2.0
    # Event types
    type_seqs = torch.randint(0, num_event_types, (batch_size, max_seq_len))
    # Sequence lengths
    seq_lens = torch.randint(5, max_seq_len + 1, (batch_size,))
    # Attention mask (batch_size, max_seq_len, max_seq_len)
    attention_mask = torch.zeros(batch_size, max_seq_len, max_seq_len, dtype=torch.bool)
    batch_non_pad_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    for i, length in enumerate(seq_lens):
        batch_non_pad_mask[i, :length] = True
        # Create causal attention mask
        for j in range(length):
            attention_mask[i, j, : j + 1] = (
                True  # Calculate time deltas (dt_seqs) - needed for some models
            )
    time_delta_seqs = torch.cat(
        [
            torch.zeros(batch_size, 1),  # First event has dt=0
            time_seqs[:, 1:]
            - time_seqs[:, :-1],  # Subsequent events have dt = t_i - t_{i-1}
        ],
        dim=1,
    )
    # Return as dictionary for easy access - only include the 5 keys that models expect
    return {
        "time_seqs": time_seqs,
        "time_delta_seqs": time_delta_seqs,
        "type_seqs": type_seqs,
        "batch_non_pad_mask": batch_non_pad_mask,
        "attention_mask": attention_mask,
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
            "time_since_start": [0.1, 0.3, 0.7, 1.2, 1.8],
            "time_since_last_event": [0.1, 0.2, 0.4, 0.5, 0.6],
            "type_event": [1, 2, 1, 3, 2],
            "idx_event": 0,
        },
        {
            "time_since_start": [0.2, 0.5, 0.9],
            "time_since_last_event": [0.2, 0.3, 0.4],
            "type_event": [2, 1, 3],
            "idx_event": 1,
        },
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
        assert (
            param.device == expected_device
        ), f"Parameter on {param.device}, expected {expected_device}"


def check_tensor_device(tensor, expected_device):
    """Check if tensor is on expected device."""
    assert (
        tensor.device == expected_device
    ), f"Tensor on {tensor.device}, expected {expected_device}"


# Skip decorators for specific conditions
skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU not available"
)

skip_if_no_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


# Custom markers
pytestmark = pytest.mark.unit
