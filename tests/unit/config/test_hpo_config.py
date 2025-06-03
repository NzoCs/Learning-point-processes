"""Unit tests for hpo_config.py configuration logic."""
import pytest
from easy_tpp.config_factory.hpo_config import HPOConfig

def test_hpo_config_defaults():
    config = HPOConfig.from_dict({
        'framework_id': None,
        'storage_uri': 'file:///tmp/hpo',
        'is_continuous': None,
        'num_trials': None,
        'num_jobs': None
    })
    assert config.framework_id == 'optuna'
    assert config.is_continuous is True
    assert config.num_trials == 50
    assert config.num_jobs == 1
    assert config.storage_uri == 'file:///tmp/hpo'

def test_hpo_config_properties():
    config = HPOConfig.from_dict({
        'framework_id': 'optuna',
        'storage_uri': 'file:///tmp/hpo',
        'is_continuous': False,
        'num_trials': 10,
        'num_jobs': 2
    })
    assert config.storage_protocol == 'file'
    assert config.storage_path == '/tmp/hpo'

def test_hpo_config_get_yaml_config():
    config = HPOConfig.from_dict({
        'framework_id': 'optuna',
        'storage_uri': 'file:///tmp/hpo',
        'is_continuous': True,
        'num_trials': 5,
        'num_jobs': 3
    })
    yaml_cfg = config.get_yaml_config()
    assert yaml_cfg['framework_id'] == 'optuna'
    assert yaml_cfg['storage_uri'] == 'file:///tmp/hpo'
    assert yaml_cfg['is_continuous'] is True
    assert yaml_cfg['num_trials'] == 5
    assert yaml_cfg['num_jobs'] == 3
