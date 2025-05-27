"""Unit tests for syn_gen_config.py configuration logic."""
import pytest
from easy_tpp.config_factory.syn_gen_config import SynGenConfig

def test_syn_gen_config_defaults():
    config = SynGenConfig({})
    assert config.num_mark == 1
    assert config.dtime_max == 5.0
    assert config.num_samples_boundary == 100
    assert config.num_samples == 20
    assert config.start_time == 0.0
    assert config.num_seq == 100
    assert config.num_batch == 10
    assert config.use_mc_samples is True
    assert config.batch_size == 32
    assert config.model_id == 'Hawkes'
    assert isinstance(config.model_config, dict)
    assert isinstance(config.sampler_config, dict)
    assert config.experiment_id == 'synthetic_data_gen'

def test_syn_gen_config_custom_fields():
    d = {'num_mark': 3, 'dtime_max': 2.5, 'num_samples': 5, 'model_id': 'RMTPP', 'experiment_id': 'exp1'}
    config = SynGenConfig(d)
    assert config.num_mark == 3
    assert config.dtime_max == 2.5
    assert config.num_samples == 5
    assert config.model_id == 'RMTPP'
    assert config.experiment_id == 'exp1'
