"""Unit tests for model_utils.py utility functions."""
import pytest
import torch
from easy_tpp.utils import model_utils

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.emb = torch.nn.Embedding(10, 3)

    def forward(self, x):
        return self.linear(x)

def test_flexible_state_dict_loading_exact_match():
    model = DummyModel()
    state_dict = model.state_dict()
    report = model_utils.flexible_state_dict_loading(model, state_dict, strict=True)
    assert 'keys_loaded' in report
    assert len(report['keys_loaded']) == len(state_dict)
    assert not report['shape_mismatch']
    assert not report['missing_keys']
    assert not report['unexpected_keys']

def test_flexible_state_dict_loading_shape_mismatch():
    model = DummyModel()
    state_dict = model.state_dict()
    # Change embedding weight shape to simulate mismatch
    state_dict['emb.weight'] = torch.randn(5, 3)  # Should be (10, 3)
    report = model_utils.flexible_state_dict_loading(model, state_dict, strict=False)
    assert 'emb.weight' in report['shape_mismatch']
    assert 'keys_loaded' in report
    assert 'missing_keys' in report
    assert 'unexpected_keys' in report

def test_flexible_state_dict_loading_unexpected_keys():
    model = DummyModel()
    state_dict = model.state_dict()
    state_dict['extra.weight'] = torch.randn(2, 2)
    report = model_utils.flexible_state_dict_loading(model, state_dict, strict=False)
    assert 'extra.weight' in report['unexpected_keys']

def test_flexible_state_dict_loading_missing_keys():
    model = DummyModel()
    state_dict = model.state_dict()
    del state_dict['linear.weight']
    report = model_utils.flexible_state_dict_loading(model, state_dict, strict=False)
    assert 'linear.weight' in report['missing_keys']
