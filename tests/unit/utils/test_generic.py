"""Unit tests for generic.py utility functions."""
import pytest
import numpy as np
from easy_tpp.utils import generic

def test_is_tensor_numpy():
    arr = np.array([1, 2, 3])
    assert generic.is_tensor(arr)
    assert generic.is_numpy_array(arr)

def test_is_tensor_torch():
    try:
        import torch
        t = torch.tensor([1, 2, 3])
        assert generic.is_tensor(t)
        assert generic.is_torch_tensor(t)
        assert generic.is_torch_device(t.device)
        assert generic.is_torch_dtype(t.dtype)
    except ImportError:
        assert not generic.is_torch_tensor(123)
        assert not generic.is_torch_device('cpu')
        assert not generic.is_torch_dtype('float32')

def test_is_tensor_tf():
    try:
        import tensorflow as tf
        t = tf.constant([1, 2, 3])
        assert generic.is_tensor(t)
        assert generic.is_tf_tensor(t)
        # Symbolic tensor test (if available)
        if hasattr(tf, 'is_symbolic_tensor'):
            symbolic = tf.raw_ops.Placeholder(dtype=tf.float32)
            assert generic.is_tf_symbolic_tensor(symbolic)
    except ImportError:
        assert not generic.is_tf_tensor(123)
        assert not generic.is_tf_symbolic_tensor(123)

def test_is_tensor_other():
    # Not a tensor
    assert not generic.is_tensor(123)
    assert not generic.is_tensor('foo')
    assert not generic.is_tensor(None)
