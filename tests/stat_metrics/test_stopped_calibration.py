"""Stopped observation contract, permutation sensitivity and independent decoder."""

import numpy as np
import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
)
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import heterogeneous_histories
from scripts.stopped_conditional_calibration import (
    empty_continuations,
    encoded_last_state,
    load_frozen_model,
    numpy_hazard,
    oracle_continuations,
)


def make_test(budget=None):
    return ConditionalMMDTwoSampleTest(
        SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2),
        99,
        stopped_event_budget=budget,
    )


def test_complete_protocol_detects_missing_events_erased_by_truncation():
    n, budget = 16, 3
    h = heterogeneous_histories(n)
    empty = empty_continuations(n, budget)
    t = torch.tensor([[0.1, 0.2, 0.3]]).repeat(n, 1)
    full = Batch(
        t,
        torch.diff(t, prepend=torch.zeros(n, 1)),
        torch.zeros_like(t, dtype=torch.long),
        torch.ones_like(t, dtype=torch.bool),
        no_next_event_mask=torch.zeros(n, dtype=torch.bool),
    )
    torch.manual_seed(53)
    historical = make_test().compute_statistics(empty, full, histories=h)
    assert historical["p_value"] == 1 and historical["observed_statistic"] == 0
    torch.manual_seed(53)
    complete = make_test(budget).compute_statistics(empty, full, histories=h)
    assert complete["p_value"] <= 0.05 and complete["observed_statistic"] > 0


def test_unknown_absence_and_inconsistent_budget_are_rejected():
    empty = empty_continuations(3, 3)
    h = heterogeneous_histories(3)
    empty.no_next_event_mask = None
    with pytest.raises(ValueError, match="boolean"):
        make_test(3).compute_statistics(empty, empty, histories=h)
    with pytest.raises(ValueError, match="ordered finite"):
        make_test().compute_statistics(empty, empty, histories=h)
    empty.no_next_event_mask = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="event budget"):
        make_test(3).compute_statistics(empty, empty, histories=h)


@pytest.mark.parametrize("name", ["FullyNN", "RMTPP"])
def test_numpy_decoder_matches_production_intensities_and_compensator(name, tmp_path):
    torch.set_num_threads(1)
    model, _ = load_frozen_model(name, tmp_path)
    h = heterogeneous_histories(2)
    states = encoded_last_state(model, h)
    sample_times = torch.tensor([[[0.0, 0.2, 1.0, 3.0]]]).repeat(2, 1, 1)
    actual = (
        model.compute_intensities_at_sample_dtimes(
            **h.to_mapping(), sample_dtimes=sample_times, compute_last_step_only=True
        )
        .detach()
        .numpy()
    )
    production, limit = model.next_event_hazard(
        **h.to_mapping(), compute_last_step_only=True
    )
    integral = production(sample_times).detach().numpy()
    for row in range(2):
        hazard, marked, maximum = numpy_hazard(model, states[row])
        np.testing.assert_allclose(
            [marked(t) for t in sample_times[row, 0].tolist()],
            actual[row, 0],
            rtol=2e-5,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            [hazard(t) for t in sample_times[row, 0].tolist()],
            integral[row, 0],
            rtol=2e-5,
            atol=5e-7,
        )
        assert maximum == pytest.approx(float(limit[row, 0]), rel=2e-5, abs=5e-7)
    first = oracle_continuations(model, h, np.random.default_rng(98), 3)
    second = oracle_continuations(model, h, np.random.default_rng(98), 3)
    for key in first.to_mapping():
        torch.testing.assert_close(
            first.to_mapping()[key], second.to_mapping()[key], rtol=0, atol=0
        )
    assert torch.equal(first.no_next_event_mask, first.valid_event_mask.sum(-1) < 3)
    assert torch.equal(first.no_next_event_mask, second.no_next_event_mask)
    forced = oracle_continuations(
        model, h, np.random.default_rng(99), 3, never_stop=True
    )
    assert forced.valid_event_mask.all() and not forced.no_next_event_mask.any()
