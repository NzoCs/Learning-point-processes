"""Independent compensator gradients, event counts and known terminal likelihoods."""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from new_ltpp.models.base.training import TrainingStepsMixin
from new_ltpp.models.capped_exponential import capped_exponential_integral
from scripts.independent_simulator_validation import (
    constant_poisson_model,
    create_model,
    history,
)


@pytest.mark.parametrize(
    "a,w,t",
    [
        (0.0, 0.0, 3.0),
        (0.0, 1e-9, 3.0),
        (0.0, -1e-9, 3.0),
        (0.0, -2.0, 50.0),
        (0.0, 1.0, 20.0),
        (12.0, -1.0, 3.0),
        (12.0, 1.0, 3.0),
        (-1000.0, 1.0, 2000.0),
        (-1000.0, -1.0, 3.0),
    ],
)
def test_capped_integral_and_gradients_against_independent_quadrature(a, w, t):
    start = torch.tensor(a, dtype=torch.double, requires_grad=True)
    slope = torch.tensor(w, dtype=torch.double, requires_grad=True)
    result = capped_exponential_integral(
        start, slope, torch.tensor(t, dtype=torch.double)
    )
    gradients = torch.autograd.grad(result, (start, slope))
    cap = math.log(1e5)
    points = [float((cap - a) / w)] if w and 0 < (cap - a) / w < t else None
    value = quad(lambda s: np.exp(min(a + w * s, cap)), 0, t, points=points)[0]
    da = quad(
        lambda s: np.exp(a + w * s) if a + w * s < cap else 0.0, 0, t, points=points
    )[0]
    dw = quad(
        lambda s: s * np.exp(a + w * s) if a + w * s < cap else 0.0, 0, t, points=points
    )[0]
    np.testing.assert_allclose(result.detach().numpy(), value, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        [g.item() for g in gradients], [da, dw], rtol=1e-8, atol=1e-8
    )


def test_zero_log_intensity_is_still_a_scored_event(tmp_path):
    mixin = TrainingStepsMixin()
    mixin.pad_token_id, mixin.eps = 2, 0.0
    mask = torch.tensor([[True, False]])
    event, _, count = mixin.compute_loglikelihood(
        torch.ones(1, 2),
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 4, 2),
        mask,
        torch.tensor([[0, 2]]),
    )
    assert event.sum() == 0 and count == 1
    model = create_model("FullyNN", tmp_path)
    model.eps = 0.0
    batch = history(2)
    batch.valid_event_mask[1, -1] = False
    batch.type_seqs[1, -1] = model.pad_token_id

    def unit_intensity(hidden, dt):
        return torch.zeros(*dt.shape, 2), torch.ones(*dt.shape, 2)

    with patch.object(model.layer_intensity, "forward", side_effect=unit_intensity):
        loss, count = model.loglike_loss(batch)
    assert loss == 0 and count == 3


def test_rmtpp_loss_matches_analytic_reference_and_does_not_draw_mc(tmp_path):
    model, _ = constant_poisson_model("RMTPP", tmp_path)
    model = model.double()
    batch = history(2)
    batch.time_seqs = batch.time_seqs.double()
    batch.time_delta_seqs = batch.time_delta_seqs.double()
    with torch.no_grad():
        model.w_t.copy_(torch.tensor([[-0.7, 0.4]], dtype=torch.double))
    with patch.object(
        model, "make_dtime_loss_samples", side_effect=AssertionError("MC")
    ):
        loss, count = model.loglike_loss(batch)
    expected = 0.0
    for row in range(2):
        for col in range(1, 3):
            dt, mark = (
                float(batch.time_delta_seqs[row, col]),
                int(batch.type_seqs[row, col]),
            )
            expected -= math.log(
                math.exp(float(model.w_t[0, mark].detach()) * dt) + model.eps
            )
            expected += sum(
                quad(lambda t: math.exp(w * t), 0, dt)[0] for w in (-0.7, 0.4)
            )
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.double))
    assert count == 4
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_tiny_growing_rate_generates_real_finite_events_against_closed_form(tmp_path):
    from new_ltpp.models.hazard_sampling import invert_hazard

    model, _ = constant_poisson_model("RMTPP", tmp_path)
    with torch.no_grad():
        model.b_t.fill_(-1000.0)
        model.w_t.fill_(1.0)
    hazard, limit = model.next_event_hazard(
        **history().to_mapping(), compute_last_step_only=True
    )
    torch.manual_seed(831)
    target = torch.empty(1, 1, 256, dtype=limit.dtype).exponential_()
    # H(t) = 2*(exp(t-1000)-exp(-1000)) before the cap. The second
    # term is below float64's representable range, so the inverse is stable.
    expected = 1000.0 + (target / 2).log()
    assert ((expected - 1000.0) < math.log(1e5)).all()
    torch.manual_seed(831)
    actual = invert_hazard(hazard, limit, 256)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("name", ["FullyNN", "RMTPP"])
def test_known_ends_add_last_valid_history_survival_cost_and_gradients(tmp_path, name):
    model = create_model(name, tmp_path)
    if name == "RMTPP":
        with torch.no_grad():
            model.w_t.fill_(-0.7)
    batch = history(2)
    batch.valid_event_mask[0, -1] = False
    batch.type_seqs[0, -1] = model.pad_token_id
    base, count = model.loglike_loss(batch)
    batch.no_next_event_mask = torch.tensor([True, False])
    complete, complete_count = model.loglike_loss(batch)
    _, limit = model.next_event_hazard(
        time_seqs=batch.time_seqs[:1, :2],
        time_delta_seqs=batch.time_delta_seqs[:1, :2],
        type_seqs=batch.type_seqs[:1, :2],
        compute_last_step_only=True,
    )
    torch.testing.assert_close(complete - base, limit.sum(), rtol=2e-5, atol=2e-6)
    assert count == complete_count == 3
    complete.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    batch.no_next_event_mask.zero_()
    torch.testing.assert_close(model.loglike_loss(batch)[0], base)


def test_impossible_known_end_is_infinite_likelihood_not_forced_parameter(tmp_path):
    model, _ = constant_poisson_model("RMTPP", tmp_path)
    batch = history()
    batch.no_next_event_mask = torch.tensor([True])
    assert torch.isinf(model.loglike_loss(batch)[0])
    with pytest.raises(ValueError, match="zero probability"):
        model.training_step(batch, 0)
    assert (model.w_t == 0).all()


@pytest.mark.parametrize("name", ["FullyNN", "RMTPP"])
def test_loss_and_gradients_ignore_padding_side_for_known_ends(tmp_path, name):
    from new_ltpp.shared_types import Batch

    model = create_model(name, tmp_path)
    if name == "RMTPP":
        with torch.no_grad():
            model.w_t.fill_(-0.5)
    source = history()
    results, gradients = [], []
    for left in (False, True):

        def padded(value, fill):
            padding = torch.full((1, 2), fill, dtype=value.dtype)
            return torch.cat((padding, value) if left else (value, padding), dim=1)

        batch = Batch(
            padded(source.time_seqs, 0),
            padded(source.time_delta_seqs, 0),
            padded(source.type_seqs, model.pad_token_id),
            padded(source.valid_event_mask, False),
            no_next_event_mask=torch.tensor([True]),
            sequence_ids=["same-history"],
        )
        before = batch.time_seqs.clone()
        loss, count = model.loglike_loss(batch)
        results.append(loss.detach())
        assert count == 2
        gradients.append(
            torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        )
        torch.testing.assert_close(batch.time_seqs, before, rtol=0, atol=0)
    torch.testing.assert_close(*results, rtol=0, atol=0)
    for a, b in zip(*gradients):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
