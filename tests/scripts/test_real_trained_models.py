import pytest
import torch

from new_ltpp.evaluation.statistical_testing.observations import configurations
from scripts.independent_simulator_validation import constant_poisson_model
from scripts.real_trained_models import fixed_window, normalize_rows
from scripts.ruche_trained_test_pilot import model_for_rates


def test_normalization_uses_one_fixed_train_scale_and_preserves_real_origin():
    rows = [dict(time_since_start=[0.0, 2.0, 6.0], type_event=[0, 1, 0])]
    normalized = normalize_rows(rows, 4.0, 2, "data", "test")[0]
    assert normalized["time_since_start"] == [0.0, 0.5, 1.5]
    assert normalized["time_since_last_event"] == [0.0, 0.5, 1.0]
    assert rows[0]["time_since_start"] == [0.0, 2.0, 6.0]


def test_fixed_window_requires_crossing_or_certified_termination(tmp_path):
    model = model_for_rates([4.0], tmp_path, "cpu")
    history = dict(times=[0.0, 0.1, 0.2, 0.3], marks=[0] * 4)
    data, proof = fixed_window(model, history, 16, 1.0, 17)
    assert proof["crossing_events"] == 16
    assert len(configurations(data, 1.0, 1)) == 16
    fast = model_for_rates([1000.0], tmp_path, "cpu")
    with pytest.raises(RuntimeError, match="did not cover"):
        fixed_window(fast, history, 16, 1.0, 17, initial=1, maximum=1)


def test_native_infinite_waiting_atom_certifies_empty_future(tmp_path):
    model, _ = constant_poisson_model("RMTPP", tmp_path)
    with torch.no_grad():
        model.w_t.fill_(-10.0)
    history = dict(times=[0.0, 0.1, 0.2, 0.3], marks=[0, 1, 0, 1])
    data, proof = fixed_window(model, history, 16, 1.0, 17, initial=8, maximum=32)
    assert proof["certified_no_next_event"] > 0
    assert (~data.valid_event_mask.any(1)).any()
    configurations(data, 1.0, 2)
