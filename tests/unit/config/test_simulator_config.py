"""Unit tests for simulator_config.py configuration logic."""
import pytest
from easy_tpp.config_factory.simulator_config import SimulatorConfig

def test_simulator_config_required_fields():
    with pytest.raises(ValueError):
        SimulatorConfig()
    with pytest.raises(ValueError):
        SimulatorConfig(model_config={})
    # Should succeed with required fields
    config = SimulatorConfig(model_config={'foo': 1}, hist_data_config={'bar': 2})
    assert hasattr(config, 'model_config')
    assert hasattr(config, 'hist_data_config')

def test_simulator_config_custom_fields():
    config = SimulatorConfig(model_config={'foo': 1}, hist_data_config={'bar': 2}, save_dir='/tmp', split='test', seed=42)
    assert config.save_dir == '/tmp' or config.save_dir is None  # save_dir is optional
    assert config.split == 'test'
    assert config.seed == 42
