"""Unit tests for misc.py utility functions."""
import os
import tempfile
import numpy as np
import pytest
import yaml
import json
from easy_tpp.utils import misc

def test_py_assert_pass():
    misc.py_assert(True, ValueError, "Should not raise")

def test_py_assert_fail():
    with pytest.raises(ValueError, match="fail message"):
        misc.py_assert(False, ValueError, "fail message")

def test_make_config_string():
    config = {'name': 'modelA', 'other': 1}
    result = misc.make_config_string(config)
    assert result.startswith('modelA')

def test_save_and_load_yaml_config():
    config = {'a': 1, 'b': 2}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.yaml')
        misc.save_yaml_config(path, config)
        loaded = misc.load_yaml_config(path)
        assert loaded == config

def test_create_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        new_folder = os.path.join(tmpdir, 'subdir')
        result = misc.create_folder(new_folder)
        assert os.path.exists(result)

def test_load_and_save_pickle():
    data = {'x': 1}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.pkl')
        misc.save_pickle(path, data)
        loaded = misc.load_pickle(path)
        assert loaded == data

def test_save_and_load_json():
    data = {'foo': [1, 2, 3]}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.json')
        misc.save_json(data, path)
        loaded = misc.load_json(path)
        assert loaded == data

def test_has_key():
    d = {'a': 1, 'b': 2}
    assert misc.has_key(d, 'a')
    assert misc.has_key(d, ['a', 'b'])
    assert not misc.has_key(d, ['a', 'c'])

def test_array_pad_cols():
    arr = np.array([[1, 2], [3, 4]])
    padded = misc.array_pad_cols(arr, 4, -1)
    assert padded.shape == (2, 4)
    assert (padded[:, :2] == arr).all()
    assert (padded[:, 2:] == -1).all()

def test_concat_element():
    arrs = [
        [np.array([[1, 2], [3, 4]])],
        [np.array([[5, 6]])]
    ]
    result = misc.concat_element(arrs, pad_index=0)
    assert isinstance(result, list)
    assert result[0].shape[0] == 3

def test_dict_deep_update():
    t = {'a': 1, 'b': {'c': 2}}
    s = {'b': {'c': 3, 'd': 4}, 'e': 5}
    updated = misc.dict_deep_update(t, s)
    assert updated['b']['c'] == 3
    assert updated['b']['d'] == 4
    assert updated['e'] == 5

def test_to_dict():
    class Dummy:
        def __init__(self):
            self.x = 1
            self.y = {'z': 2}
    d = Dummy()
    result = misc.to_dict(d)
    assert isinstance(result, dict)
    assert 'x' in result
    assert 'y' in result
