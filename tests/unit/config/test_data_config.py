"""
Unit tests for the refactored DataConfig, TokenizerConfig, and DataLoadingSpecsConfig.
"""

import pytest

from easy_tpp.configs.base_config import ConfigValidationError
from easy_tpp.configs.data_config import (
    DataConfig,
    DataLoadingSpecsConfig,
    TokenizerConfig,
)


def test_tokenizer_config_defaults():
    config = TokenizerConfig(num_event_types=5)
    assert config.num_event_types == 5
    assert config.pad_token_id == 5
    assert config.num_event_types_pad == 6
    assert config.padding_side == "left"
    assert config.truncation_side == "left"
    assert config.padding_strategy == "longest"
    assert config.max_len is None


def test_tokenizer_config_invalid_padding():
    with pytest.raises(ConfigValidationError):
        TokenizerConfig(num_event_types=5, padding_side="center")


def test_tokenizer_config_invalid_truncation():
    with pytest.raises(ConfigValidationError):
        TokenizerConfig(num_event_types=5, truncation_side="center")


def test_tokenizer_config_from_dict():
    d = {"num_event_types": 3, "padding_side": "right"}
    config = TokenizerConfig.from_dict(d)
    assert config.num_event_types == 3
    assert config.padding_side == "right"


def test_data_loading_specs_config_defaults():
    config = DataLoadingSpecsConfig()
    assert config.batch_size == 32
    assert config.num_workers == 1
    assert config.tensor_type == "pt"
    assert config.max_len is None


def test_data_loading_specs_config_from_dict():
    d = {"batch_size": 64, "tensor_type": "tf"}
    config = DataLoadingSpecsConfig.from_dict(d)
    assert config.batch_size == 64
    assert config.tensor_type == "tf"


def test_data_config_minimal():
    config = DataConfig(
        train_dir="train.csv", data_specs=TokenizerConfig(num_event_types=2)
    )
    assert config.train_dir == "train.csv"
    assert config.data_specs.num_event_types == 2
    assert config.data_format == "csv"


def test_data_config_from_dict():
    d = {
        "train_dir": "train.csv",
        "data_loading_specs": {"batch_size": 16},
        "data_specs": {"num_event_types": 4},
    }
    config = DataConfig.from_dict(d)
    assert config.train_dir == "train.csv"
    assert config.data_loading_specs.batch_size == 16
    assert config.data_specs.num_event_types == 4


def test_data_config_get_data_dir():
    config = DataConfig(
        train_dir="train.csv",
        valid_dir="valid.csv",
        test_dir="test.csv",
        source_dir="all.csv",
        data_specs=TokenizerConfig(num_event_types=2),
    )
    assert config.get_data_dir("train") == "train.csv"
    assert config.get_data_dir("valid") == "valid.csv"
    assert config.get_data_dir("test") == "test.csv"
    assert config.get_data_dir(None) == "all.csv"
    with pytest.raises(ValueError):
        config.get_data_dir("unknown")


def test_data_config_missing_source():
    config = DataConfig(data_specs=TokenizerConfig(num_event_types=2))
    with pytest.raises(ValueError):
        config.get_data_dir(None)
