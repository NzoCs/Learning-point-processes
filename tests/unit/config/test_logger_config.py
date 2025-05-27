"""Unit tests for logger_config.py configuration logic."""
import pytest
from easy_tpp.config_factory import logger_config

class DummyLoggerAdapter(logger_config.BaseLoggerAdapter):
    @classmethod
    def get_required_params(cls):
        return ['foo', 'bar']
    @classmethod
    def configure(cls, config):
        return config

def test_logger_type_enum():
    assert logger_config.LoggerType.CSV.value == 'csv'
    assert logger_config.LoggerType.WandB.value == 'wandb'
    assert logger_config.LoggerType.TENSORBOARD.value == 'tensorboard'

def test_base_logger_adapter_required_params():
    assert DummyLoggerAdapter.get_required_params() == ['foo', 'bar']

def test_base_logger_adapter_validate_config_success():
    config = {'foo': 1, 'bar': 2}
    validated = DummyLoggerAdapter.validate_config(config)
    assert validated == config

def test_base_logger_adapter_validate_config_missing():
    config = {'foo': 1}
    with pytest.raises(ValueError):
        DummyLoggerAdapter.validate_config(config)
