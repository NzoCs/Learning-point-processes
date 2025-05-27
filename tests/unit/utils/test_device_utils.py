"""Unit tests for device_utils.py utility functions."""
import pytest
import torch
from easy_tpp.utils import device_utils

def test_ensure_same_device_cpu():
    t1 = torch.randn(2, 2)
    t2 = torch.ones(2, 2)
    out1, out2 = device_utils.ensure_same_device(t1, t2)
    assert out1.device == out2.device == t1.device
    assert torch.allclose(out2, torch.ones(2, 2))

def test_ensure_same_device_with_target():
    t1 = torch.randn(2, 2)
    t2 = torch.ones(2, 2)
    out1, out2 = device_utils.ensure_same_device(t1, t2, target_device='cpu')
    assert out1.device.type == 'cpu' and out2.device.type == 'cpu'

def test_ensure_same_device_none_and_non_tensor():
    t1 = torch.randn(2, 2)
    out = device_utils.ensure_same_device(t1, None, 'not_a_tensor')
    assert isinstance(out, tuple)
    assert out[0].device == t1.device
    assert out[1] is None
    assert out[2] == 'not_a_tensor'

def test_ensure_same_device_empty():
    assert device_utils.ensure_same_device() == ()

def test_get_device_info_keys():
    info = device_utils.get_device_info()
    assert 'cuda_available' in info
    assert 'cuda_device_count' in info
    assert 'current_device' in info

def test_clear_gpu_cache_runs():
    # Should not raise, even if no GPU
    device_utils.clear_gpu_cache()
