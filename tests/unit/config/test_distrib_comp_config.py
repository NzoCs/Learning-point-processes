"""Unit tests for distrib_comp_config.py configuration logic."""
import pytest
import os
from easy_tpp.config_factory.distrib_comp_config import DistribCompConfig

def test_distrib_comp_config_defaults(tmp_path):
    config = DistribCompConfig(output_dir=str(tmp_path))
    assert os.path.exists(config.output_dir)
    assert config.label_split == 'test'
    assert config.pred_split is None
    assert isinstance(config.data_specs, dict)

def test_distrib_comp_config_custom_fields(tmp_path):
    config = DistribCompConfig(
        output_dir=str(tmp_path),
        label_split='valid',
        pred_split='train',
        data_specs={'foo': 1},
        label_data_config={'bar': 2},
        pred_data_config={'baz': 3},
        data_loading_specs={'batch_size': 4}
    )
    assert config.label_split == 'valid'
    assert config.pred_split == 'train'
    assert config.data_specs['foo'] == 1
    assert config.label_data_config['bar'] == 2
    assert config.pred_data_config['baz'] == 3
    assert config.data_loading_specs['batch_size'] == 4

def test_distrib_comp_config_dynamic_fields(tmp_path):
    config = DistribCompConfig(output_dir=str(tmp_path), custom_field=123)
    assert hasattr(config, 'custom_field')
    assert config.custom_field == 123
