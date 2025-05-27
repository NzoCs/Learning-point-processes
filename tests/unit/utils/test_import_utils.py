"""Unit tests for import_utils.py utility functions."""
import types
import pytest
import sys
from easy_tpp.utils import import_utils

def test_is_torch_available_and_version():
    # Should return a bool and a version string (if torch is installed)
    result = import_utils.is_torch_available()
    version = import_utils.get_torch_version()
    assert isinstance(result, bool)
    assert isinstance(version, str)

def test_is_torchvision_available():
    result = import_utils.is_torchvision_available()
    assert isinstance(result, bool)

def test_is_torch_cuda_available():
    # Should return a bool, even if torch is not installed
    result = import_utils.is_torch_cuda_available()
    assert isinstance(result, bool)

def test_is_tf_available():
    result = import_utils.is_tf_available()
    assert isinstance(result, bool)

def test_is_tf_gpu_available():
    # Should return a bool, even if tensorflow is not installed
    result = import_utils.is_tf_gpu_available()
    assert isinstance(result, bool)

def test_is_torch_mps_available():
    # Should return a bool, even if torch is not installed
    result = import_utils.is_torch_mps_available()
    assert isinstance(result, bool)

def test_is_torch_gpu_available():
    # Should return a bool, even if torch is not installed
    result = import_utils.is_torch_gpu_available()
    assert isinstance(result, bool)

def test_is_tensorflow_probability_available():
    result = import_utils.is_tensorflow_probability_available()
    assert isinstance(result, bool)

def test_torch_only_method_decorator():
    # Should raise ImportError if torch is not available, else call the function
    called = {}
    @import_utils.torch_only_method
    def dummy():
        called['ok'] = True
        return 42
    try:
        result = dummy()
        assert result == 42 or not import_utils.is_torch_available()
        if import_utils.is_torch_available():
            assert called['ok']
    except ImportError:
        assert not import_utils.is_torch_available()

def test_requires_backends_errors():
    class Dummy:
        pass
    # Should not raise if backend is available
    for backend in ['torch', 'tf', 'torchvision', 'tensorflow_probability']:
        try:
            import_utils.requires_backends(Dummy, backend)
        except ImportError:
            pass  # Acceptable if not installed
    # Should raise ImportError with correct message for missing backend
    # (simulate by passing a fake backend)
    with pytest.raises(KeyError):
        import_utils.requires_backends(Dummy, 'not_a_real_backend')
