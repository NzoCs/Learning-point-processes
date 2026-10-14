"""Unbounded delays, distribution preservation and explicit sampling safety."""

import math

import numpy as np
import pytest
import torch
from scipy.stats import kstest

from new_ltpp.configs.model_config import ThinningConfig
from new_ltpp.models.event_sampler import EventSampler, SamplingSafetyError


def sampler(*, window=0.25, rounds=10000, proposals=32, device="cpu"):
    return EventSampler(proposals, 2.0, 8, window, torch.device(device), rounds)


def draw(engine, intensity, size=1):
    t = torch.zeros(size, 1, device=engine.device)
    return engine.draw_next_time_one_step(
        t, t.clone(), t.long(), t.bool(), intensity, 1
    )


def constant(rate):
    return lambda sample_dtimes, **kwargs: torch.full_like(
        sample_dtimes, rate
    ).unsqueeze(-1)


@pytest.mark.parametrize("rate", [0.2, 2.0])
def test_long_waits_follow_poisson_without_an_atom_at_old_cap(rate):
    torch.set_num_threads(1)
    torch.manual_seed(20261021)
    delays, weights = draw(sampler(), constant(rate), 4096)
    values = delays.flatten().numpy()
    bound = math.sqrt(math.log(2 / 1e-4) / (2 * len(values)))
    assert kstest(values * rate, "expon").statistic <= bound
    assert (values > 0.25).any() and not (values == 0.25).any()
    assert torch.isfinite(delays).all() and (weights == 1).all()


def test_rising_intensity_recomputes_local_envelopes_and_preserves_compensator():
    torch.manual_seed(20261022)
    intensity = lambda sample_dtimes, **kwargs: (0.5 + 0.3 * sample_dtimes).unsqueeze(
        -1
    )
    delays, _ = draw(sampler(), intensity, 4096)
    compensator = (0.5 * delays + 0.15 * delays.square()).flatten().numpy()
    assert kstest(compensator, "expon").statistic <= math.sqrt(
        math.log(2 / 1e-4) / (2 * len(compensator))
    )


def test_sampling_continues_past_ten_rejected_rounds(monkeypatch):
    engine = sampler(window=100, proposals=1, rounds=14)
    monkeypatch.setattr(
        engine,
        "compute_intensity_upper_bound",
        lambda *args, **kwargs: torch.ones(1, 1),
    )
    monkeypatch.setattr(
        engine, "sample_exp_distribution", lambda rate: torch.full((1, 1, 1), 0.1)
    )
    calls = []

    def uniforms(rate, count):
        calls.append(True)
        return torch.full((1, 1, count, 1), 0.0 if len(calls) == 13 else 1.0)

    monkeypatch.setattr(engine, "sample_uniform", uniforms)
    actual, _ = draw(engine, constant(0.01))
    assert len(calls) == 13 and actual.item() == pytest.approx(1.3)


def test_zero_intensity_reaches_budget_without_fabricating_an_event():
    torch.manual_seed(43)
    with pytest.raises(SamplingSafetyError, match="proposal budget exhausted") as error:
        draw(sampler(rounds=4), constant(0), 3)
    assert error.value.rounds == 4 and error.value.pending_count == 3
    assert error.value.state["pending"].all()
    assert not error.value.state["accepted_dtimes"].any()
    assert (error.value.state["offset"] > 0.25).all()


def test_invalid_envelope_is_reported_before_returning_a_biased_draw(monkeypatch):
    engine = sampler(window=100)
    monkeypatch.setattr(
        engine,
        "compute_intensity_upper_bound",
        lambda *args, **kwargs: torch.full((1, 1), 0.1),
    )
    monkeypatch.setattr(
        engine, "sample_exp_distribution", lambda rate: torch.full((1, 1, 32), 0.1)
    )
    with pytest.raises(SamplingSafetyError, match="exceeds the local envelope"):
        draw(engine, constant(1))


def test_nonfinite_intensity_stops_safely():
    with pytest.raises(SamplingSafetyError, match="nonfinite"):
        draw(sampler(), constant(float("nan")))


def test_negative_mark_intensity_is_not_hidden_by_a_positive_total():
    def intensity(sample_dtimes, **kwargs):
        return torch.stack(
            (torch.full_like(sample_dtimes, -0.1), torch.full_like(sample_dtimes, 0.2)),
            -1,
        )

    with pytest.raises(SamplingSafetyError, match="negative intensity"):
        draw(sampler(), intensity)


def test_floating_point_stagnation_stops_safely(monkeypatch):
    engine = sampler()
    monkeypatch.setattr(
        engine, "sample_exp_distribution", lambda rate: torch.zeros(1, 1, 32)
    )
    with pytest.raises(SamplingSafetyError, match="stagnation"):
        draw(engine, constant(1))


def test_complete_sampling_repeats_exactly_with_seed():
    engine = sampler()
    torch.manual_seed(44)
    first, _ = draw(engine, constant(0.2), 64)
    torch.manual_seed(44)
    second, _ = draw(engine, constant(0.2), 64)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_rmtpp_negative_slopes_have_a_nonzero_probability_of_no_next_event(tmp_path):
    from scripts.independent_simulator_validation import constant_poisson_model

    model, _ = constant_poisson_model("RMTPP", tmp_path)
    with torch.no_grad():
        model.w_t.fill_(-1)
    times = torch.linspace(0, 10, 101).reshape(1, 1, -1)
    values = model.evolve_and_get_intentsity(
        torch.zeros(1, 1, model.hidden_size), times
    ).sum(-1)
    torch.testing.assert_close(values, 2 * (-times).exp())
    # Integral to infinity of 2 exp(-t) equals 2, leaving mass exp(-2) at infinity.
    assert math.exp(-2) == pytest.approx(0.1353352832366127)


@pytest.mark.parametrize("rounds", [0, -1, None])
def test_safety_budget_configuration_requires_a_positive_integer(rounds):
    with pytest.raises(ValueError):
        ThinningConfig(num_sample=1, num_exp=1, max_sampling_rounds=rounds)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_unbounded_sampling_keeps_results_on_gpu():
    delays, weights = draw(sampler(device="cuda"), constant(0.2), 128)
    assert delays.is_cuda and weights.is_cuda and torch.isfinite(delays).all()
