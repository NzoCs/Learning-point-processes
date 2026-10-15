"""Dedicated generation is the sole parametric pipeline path; no unused neural layers."""

import numpy as np
import pytest
import torch

from new_ltpp.models.base.base_model import NeuralModel
from new_ltpp.models.base.parametric import ParametricModel
from scripts.independent_simulator_validation import create_model, history


@pytest.mark.parametrize("name", ["Hawkes", "SelfCorrecting", "Poisson"])
def test_parametric_models_have_no_neural_embedding_or_thinning(tmp_path, name):
    model = create_model(name, tmp_path)
    assert isinstance(model, ParametricModel) and not isinstance(model, NeuralModel)
    assert not hasattr(model, "layer_type_emb") and not hasattr(
        model, "get_event_sampler"
    )
    assert not hasattr(model, "hidden_size")
    assert all("layer_type_emb" not in key for key in model.state_dict())
    first = model.simulate(history(2), 0)
    assert first.time_seqs.shape == (2, 0)
    torch.manual_seed(53)
    result = model.simulate(history(2), 4)
    assert result.valid_event_mask.all() and torch.isfinite(result.time_seqs).all()
    torch.testing.assert_close(
        result.time_seqs, result.time_delta_seqs.cumsum(1), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("name", ["Hawkes", "SelfCorrecting", "Poisson"])
def test_parametric_prediction_is_causal_and_simulation_repeatable(tmp_path, name):
    model = create_model(name, tmp_path)
    batch = history(2)
    batch.sequence_ids = ["a", "b"]
    torch.manual_seed(53)
    first = model.simulate(batch, 4)
    torch.manual_seed(53)
    second = model.simulate(batch, 4)
    assert first.sequence_ids == batch.sequence_ids
    for key, value in first.to_mapping().items():
        torch.testing.assert_close(value, second.to_mapping()[key], rtol=0, atol=0)
    torch.manual_seed(54)
    first_prediction = model.predict_one_step_at_every_event(**batch.to_mapping())
    altered = history(2)
    altered.time_seqs[:, -1] = 99
    altered.type_seqs[:, -1] = 1
    torch.manual_seed(54)
    second_prediction = model.predict_one_step_at_every_event(**altered.to_mapping())
    torch.testing.assert_close(
        first_prediction["dtime_predict"],
        second_prediction["dtime_predict"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first_prediction["type_predict"],
        second_prediction["type_predict"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("name", ["Hawkes", "SelfCorrecting", "Poisson"])
def test_horizon_generation_preserves_padding_and_stops_at_horizon(tmp_path, name):
    model = create_model(name, tmp_path)
    result = model.simulate_until(8, 2.0)
    assert result.time_seqs.shape == result.valid_event_mask.shape
    assert (result.time_seqs[result.valid_event_mask] < 2).all()
    assert torch.isfinite(result.time_delta_seqs).all()
    assert (result.type_seqs[~result.valid_event_mask] == model.pad_token_id).all()


@pytest.mark.parametrize("name", ["Hawkes", "SelfCorrecting", "Poisson"])
def test_intensity_queries_agree_with_native_post_history_state(tmp_path, name):
    model = create_model(name, tmp_path)
    batch = history(2)
    delays = torch.full((*batch.time_seqs.shape, 1), 0.4)
    expected = model.continuation_intensities(batch, delays[:, -1])
    for last_only in (False, True):
        actual = model.compute_intensities_at_sample_dtimes(
            **batch.to_mapping(), sample_dtimes=delays, compute_last_step_only=last_only
        )
        torch.testing.assert_close(actual[:, -1], expected, rtol=1e-6, atol=1e-7)


def test_self_correcting_likelihood_conditions_on_first_event(tmp_path):
    from new_ltpp.shared_types import Batch
    from scripts.independent_simulator_validation import ALPHA, MU

    model = create_model(
        "SelfCorrecting", tmp_path, mu=MU.tolist(), alpha=ALPHA.tolist()
    )
    times = torch.tensor([[0.1, 0.3]])
    batch = Batch(
        times,
        torch.tensor([[0.1, 0.2]]),
        torch.tensor([[0, 1]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    actual, count = model.loglike_loss(batch)
    penalty = ALPHA[:, 0]
    expected = np.sum(np.exp(MU * 0.1 - penalty) * np.expm1(MU * 0.2) / MU) - (
        MU[1] * 0.3 - penalty[1]
    )
    assert count == 1 and actual.item() == pytest.approx(expected, rel=1e-6)


def test_self_correcting_large_history_does_not_overflow_inverse(tmp_path, monkeypatch):
    from new_ltpp.shared_types import Batch

    model = create_model(
        "SelfCorrecting", tmp_path, mu=[0.2, 0.3], alpha=[[1, 1], [1, 1]]
    )
    times = torch.linspace(0, 1, 300).unsqueeze(0)
    batch = Batch(
        times,
        torch.diff(times, prepend=torch.zeros(1, 1)),
        torch.zeros(1, 300, dtype=torch.long),
        torch.ones(1, 300, dtype=torch.bool),
    )
    monkeypatch.setattr(
        torch.Tensor, "exponential_", lambda tensor, *args, **kwargs: tensor.fill_(1)
    )
    actual = model.simulate(batch, 1).time_delta_seqs.item()
    expected = min(np.logaddexp(0, np.log(mu) + 300 - mu) / mu for mu in (0.2, 0.3))
    assert np.isfinite(actual) and actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA hardware required; CPU is not a GPU proxy",
)
@pytest.mark.parametrize("name", ["Hawkes", "SelfCorrecting", "Poisson"])
def test_parametric_simulation_keeps_all_results_on_cuda(tmp_path, name):
    model = create_model(name, tmp_path).cuda()
    result = model.simulate(history(128).to_device(torch.device("cuda")), 16)
    assert all(value.is_cuda for value in result.to_mapping().values())
    assert torch.isfinite(result.time_seqs).all() and result.valid_event_mask.all()
