"""Public numerical contracts, independently of the historical snapshots."""

import json
from pathlib import Path

import pytest
import torch

from new_ltpp.configs import ModelConfig
from new_ltpp.models.model_factory import ModelFactory
from new_ltpp.shared_types import Batch

MODELS = [
    "ANHN",
    "ANHP",
    "FullyNN",
    "Hawkes",
    "IntensityFree",
    "NHP",
    "ODETPP",
    "RMTPP",
    "SAHP",
    "SelfCorrecting",
    "THP",
]


@pytest.fixture
def batch():
    times = torch.tensor([[0.0, 0.2, 0.7, 1.3], [0.0, 0.3, 0.8, 1.4]])
    deltas = torch.cat([times[:, :1], times[:, 1:] - times[:, :-1]], dim=1)
    return Batch(
        times,
        deltas,
        torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
        torch.ones_like(times, dtype=torch.bool),
    )


@pytest.fixture(params=MODELS)
def model(request, tmp_path):
    import new_ltpp.models  # noqa: F401

    torch.set_num_threads(1)
    torch.manual_seed(123)
    raw = json.loads((Path(__file__).parent / "integration/config.json").read_text())[
        "model_config"
    ]
    raw["model_id"] = request.param
    return ModelFactory.create_model_by_name(
        request.param,
        ModelConfig.model_validate(raw),
        {
            "num_event_types": 2,
            "end_time_max": 2.0,
            "dtime_max": 1.0,
            "pad_token_id": 2,
        },
        tmp_path,
    )


def test_likelihood_has_finite_gradients_and_can_be_optimized(model, batch):
    loss, count = model.loglike_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert 0 < count <= batch.valid_event_mask.sum()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_prediction_has_valid_times_and_marks_and_repeats_with_seed(model, batch):
    model.eval()
    torch.manual_seed(91)
    first = model.predict_one_step_at_every_event(**batch.to_mapping())
    torch.manual_seed(91)
    second = model.predict_one_step_at_every_event(**batch.to_mapping())
    for key in ("dtime_predict", "type_predict"):
        assert first[key].shape == (2, 3)
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert torch.isfinite(first["dtime_predict"]).all()
    assert (first["dtime_predict"] >= 0).all()
    assert ((first["type_predict"] >= 0) & (first["type_predict"] < 2)).all()


def test_intensity_is_finite_nonnegative_and_has_expected_axes(model, batch):
    if not getattr(model, "supports_intensity", True):
        pytest.skip("Density model has no intensity API")
    values = model.compute_intensities_at_sample_dtimes(
        **batch.to_mapping(), sample_dtimes=torch.full((2, 4, 3), 0.1)
    )
    assert values.shape == (2, 4, 3, 2)
    assert torch.isfinite(values).all() and (values >= 0).all()
