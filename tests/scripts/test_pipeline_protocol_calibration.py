"""Observation and independent-oracle controls for the opt-in campaign."""

import numpy as np
import pytest
import torch

from scripts.pipeline_protocol_calibration import (
    SCENARIOS,
    analytic_continuations,
    evaluate,
    histories_for_trial,
    holdout,
    model_and_config,
    observed_future,
    poisson_continuations,
    poisson_control,
    state_hash,
)


def test_quotas_and_singleton_tail_match_declared_design():
    assert SCENARIOS == {"quota3": (3, (16, 11, 5)), "quota6": (6, (64, 63, 1))}


def test_holdout_retains_empty_and_terminal_future_flags():
    h = histories_for_trial(3, np.random.default_rng(1), 7)
    x = poisson_continuations(h, np.random.default_rng(2), 3, np.ones(2))
    x.no_next_event_mask = torch.tensor([True, True, False])
    x.valid_event_mask[0] = False
    x.valid_event_mask[1, 1:] = False
    x.time_seqs.masked_fill_(~x.valid_event_mask, 0)
    x.type_seqs.masked_fill_(~x.valid_event_mask, 2)
    prefix, future = holdout(h, x, 3, True)
    assert future.sequence_ids == prefix.sequence_ids == h.sequence_ids
    assert future.valid_event_mask.sum(1).tolist() == [0, 1, 3]
    assert future.no_next_event_mask.tolist() == [True, True, False]
    torch.testing.assert_close(prefix.time_seqs, h.time_seqs, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Not enough valid"):
        holdout(h, x, 3, False)


def test_independent_rmtpp_oracle_recovers_known_marked_poisson(tmp_path):
    from scipy.stats import kstest

    model, _, fit = model_and_config("RMTPP", tmp_path)
    assert fit["effective_model_config"]["thinning_config"]["over_sample_rate"] == 1.1
    with torch.no_grad():
        model.hidden_to_intensity_logits.weight.zero_()
        model.hidden_to_intensity_logits.bias.zero_()
        model.b_t.zero_()
        model.w_t.zero_()
    h = histories_for_trial(512, np.random.default_rng(3), 0)
    x = analytic_continuations(model, h, np.random.default_rng(4), 3)
    assert x.valid_event_mask.all() and not x.no_next_event_mask.any()
    assert kstest((x.time_delta_seqs * 2).numpy().ravel(), "expon").statistic < 0.06
    assert abs(float((x.type_seqs == 0).float().mean()) - 0.5) < 0.05


def test_current_accumulator_matches_dense_and_replays_with_terminal_rows(tmp_path):
    model, config, _ = model_and_config("FullyNN", tmp_path)
    h = histories_for_trial(5, np.random.default_rng(5), 0)
    x = observed_future(model, h, np.random.default_rng(6), 3, 7)
    before = state_hash(model)
    a, _, _ = evaluate(model, config, h, x, (2, 2, 1), 3, 8, dense_check=True)
    b, _, _ = evaluate(model, config, h, x, (2, 2, 1), 3, 8)
    assert a == b and len(a["global_pair_ids"]) == 5
    assert len(a["global_permuted_statistic"]) == 99
    assert state_hash(model) == before


@pytest.mark.parametrize("name", ["ANHN", "ANHP", "NHP", "ODETPP", "SAHP", "THP"])
def test_poisson_control_preserves_fitted_model_and_sampler_settings(tmp_path, name):
    model, _, _ = model_and_config(name, tmp_path)
    before = state_hash(model)
    control = poisson_control(model)
    h = histories_for_trial(2, np.random.default_rng(9), 0)
    actual = control.compute_intensities_at_sample_dtimes(
        **h.to_mapping(),
        sample_dtimes=torch.tensor([[[0.1, 1.0]], [[0.2, 2.0]]]).expand(2, 2, 2),
        compute_last_step_only=True,
    )
    torch.testing.assert_close(
        actual, torch.full_like(actual, np.log(2)), rtol=1e-6, atol=1e-6
    )
    assert state_hash(model) == before
    for field in (
        "num_exp",
        "num_samples_boundary",
        "over_sample_rate",
        "max_sampling_rounds",
        "num_mc_samples",
    ):
        assert getattr(control, field) == getattr(model, field)
