"""Scientific design invariants, independent oracle and replay checks."""

import itertools
import json

import numpy as np
import pytest
import torch

from new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test import (
    paired_joint_statistic,
)
from scripts.report_experiment_summary import aggregate, holm_adjust
from scripts.report_experiment_validation import (
    CASES,
    EXP,
    POISSON,
    Process,
    event_hash,
    method_grams,
    pilot,
    prepare_paths,
    read_records,
    run_cell,
    seed_entropy,
    simulate_events,
    summarize,
    swap_statistics,
    uniformize_counts,
)


@pytest.mark.parametrize(
    "process",
    [Process("unknown"), Process("exp", branching=1), Process("power", exponent=1)],
)
def test_oracle_rejects_invalid_process(process):
    with pytest.raises(ValueError):
        simulate_events(np.random.default_rng(3), process, 2)


def test_oracle_has_empty_paths_and_never_truncates_on_budget_failure():
    rows = simulate_events(
        np.random.default_rng(1), Process("exp", rate=1e-20), 3, burn=0
    )
    assert len(rows) == 3 and all(len(r) == 0 for r in rows)
    with pytest.raises(RuntimeError, match="budget"):
        simulate_events(np.random.default_rng(2), POISSON, 100, max_events=1)


def test_independent_poisson_moments_and_uniform_conditional_locations():
    rows = simulate_events(np.random.default_rng(3), Process("poisson", rate=4), 4000)
    counts = np.array([len(r) for r in rows])
    assert abs(counts.mean() - 4) < 0.15
    assert abs(counts.var() - 4) < 0.3
    locations = np.concatenate(rows)
    assert abs(locations.mean() - 0.5) < 0.01
    assert abs((locations < 0.25).mean() - 0.25) < 0.015


def test_exponential_hawkes_stationary_count_mean_and_excess_dispersion():
    rows = simulate_events(np.random.default_rng(4), EXP, 2000)
    counts = np.array([len(r) for r in rows])
    assert abs(counts.mean() - 100) < 2
    assert counts.var() > 200  # Poisson rate 100 would have count variance 100.
    assert all(np.all(np.diff(r) >= 0) and np.all((r > 0) & (r < 1)) for r in rows)


def test_timing_control_preserves_each_count_and_does_not_mutate_inputs():
    rows = [np.array([]), np.array([0.1]), np.array([0.1, 0.2, 0.25])]
    before = event_hash(rows)
    transformed = uniformize_counts(rows, np.random.default_rng(5), 1.0)
    assert list(map(len, rows)) == list(map(len, transformed))
    assert event_hash(rows) == before
    assert not np.array_equal(rows[-1], transformed[-1])
    assert all(np.all(np.diff(r) >= 0) for r in transformed)


def test_pilot_is_shared_between_null_and_alternative_and_independent_of_evaluation():
    a, pa = pilot("poisson_H0", 10, 1.0, 10.0)
    b, pb = pilot("poisson_rate_110", 10, 1.0, 10.0)
    assert a == b and pa == pb
    entropy = [
        seed_entropy(case, cohort, trial, stream)
        for case in CASES
        for cohort in (0, 1)
        for trial in range(5)
        for stream in (1, 2, 3)
    ]
    assert len(set(entropy)) == len(entropy)
    assert tuple(pa["entropy"]) not in entropy
    assert seed_entropy("poisson_H0", 0, 0, 1, domain=0) != seed_entropy(
        "poisson_H0", 0, 0, 1, domain=1
    )


def test_count_baseline_is_exactly_blind_to_count_matched_time_changes():
    x = [np.array([0.1, 0.2]), np.array([0.7]), np.array([])]
    y = uniformize_counts(x, np.random.default_rng(6), 1.0)
    grams = dict(method_grams(prepare_paths(x + y, 2.0, 8), "cpu"))
    signs = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)))
    values = swap_statistics(grams["count_rbf"], signs, verify=True)
    assert torch.equal(values, torch.zeros_like(values))


def test_vectorization_matches_public_statistic_on_every_swap():
    points = torch.randn(
        6, 4, generator=torch.Generator().manual_seed(8), dtype=torch.float64
    )
    gram = points @ points.T
    signs = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)))
    actual = swap_statistics(gram, signs)
    rows = torch.arange(3)
    expected = []
    for s in signs:
        swap = (s < 0).long()
        expected.append(
            paired_joint_statistic(gram, rows + 3 * swap, rows + 3 * (1 - swap))
        )
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)


def test_checkpoint_retains_torn_bytes_and_rejects_duplicate_trials(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_bytes(b'{"trial":0}\n{"tri')
    assert read_records(path) == [{"trial": 0}]
    assert path.with_suffix(".torn").read_bytes() == b'{"tri'
    path.write_text('{"trial":0}\n{"trial":0}\n')
    with pytest.raises(ValueError, match="unique"):
        read_records(path)


def test_failed_draws_cannot_be_reported_as_calibrated():
    stats = summarize("poisson_H0", [{"status": "failed", "trial": 0}], 1)
    assert all(
        not r["calibration_assessable"] and r["valid_trials"] == 0 for r in stats
    )


def test_small_actual_backend_cell_replays_and_refuses_mixed_design(tmp_path):
    torch.set_num_threads(1)
    first = run_cell(
        "poisson_H0", 0, tmp_path / "a", trials=2, n=4, permutations=19, grid=8
    )
    second = run_cell(
        "poisson_H0", 0, tmp_path / "b", trials=2, n=4, permutations=19, grid=8
    )
    assert first["results"] == second["results"]
    assert (tmp_path / "a/trials.jsonl").read_bytes() == (
        tmp_path / "b/trials.jsonl"
    ).read_bytes()
    saved = (tmp_path / "a/trials.jsonl").read_bytes()
    run_cell("poisson_H0", 0, tmp_path / "a", trials=2, n=4, permutations=19, grid=8)
    assert (tmp_path / "a/trials.jsonl").read_bytes() == saved
    with pytest.raises(ValueError, match="changed protocol"):
        run_cell(
            "poisson_H0", 0, tmp_path / "a", trials=3, n=4, permutations=19, grid=8
        )
    plan = json.loads((tmp_path / "a/plan.json").read_text())
    assert plan["same_counts"] is False and plan["pilot"]["paths"] == 512


def test_incomplete_campaign_retains_errors_and_cannot_pass(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregate(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert len(summary["errors"]) == 20
    assert summary["precision_sufficient_for_all_pooled_cells"] is False


def test_multiplicity_correction_keeps_prespecified_family_if_results_missing():
    assert holm_adjust([0.01, 0.03, 0.2]) == pytest.approx([0.03, 0.06, 0.2])
    assert holm_adjust([0.01], family_size=18) == pytest.approx([0.18])
    with pytest.raises(ValueError):
        holm_adjust([0.01, 0.02], family_size=1)
