"""Independent mathematical checks and exact MMD/RMSE replay contracts."""

import json

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from scripts.report_experiment_validation import (
    EXP,
    POISSON,
    Process,
    run_cell,
    simulate_events,
)
from scripts.report_rmse_validation import (
    aggregate,
    cumulative_hazard,
    gap_rmse,
    landmark_losses,
    loss_test,
    regenerate,
    replay,
    waiting_mean,
)


@pytest.mark.parametrize(
    "process", [EXP, Process("power", exponent=1.5), Process("power", exponent=3)]
)
def test_hazard_matches_direct_integral_of_original_intensity(process):
    history, time = np.array([0.03, 0.12, 0.2]), 0.2
    mu = process.rate * (1 - process.branching)
    alpha = process.branching * process.beta
    if process.kind == "power":
        alpha *= process.exponent - 1

    def intensity(u):
        age = time + u - history
        kernel = (
            np.exp(-process.beta * age)
            if process.kind == "exp"
            else (1 + process.beta * age) ** (-process.exponent)
        )
        return mu + alpha * kernel.sum()

    for delay in (0.001, 0.1, 1):
        expected = quad(intensity, 0, delay, epsabs=1e-11)[0]
        assert cumulative_hazard(np.array([delay]), history, time, process)[
            0
        ] == pytest.approx(expected, rel=1e-10)
    restricted = quad(lambda u: np.exp(-quad(intensity, 0, u)[0]), 0, 0.25)[0]
    assert waiting_mean(history, time, process, 0.25) == pytest.approx(
        restricted, rel=1e-9
    )


def test_poisson_restricted_mean_and_exp_current_event_included():
    cap, rate = 0.03, 100
    expected = (1 - np.exp(-rate * cap)) / rate
    assert waiting_mean([], 0, POISSON, cap) == expected
    assert waiting_mean([0.2], 0.2, EXP) < waiting_mean([], 0.2, EXP)
    with pytest.raises(ValueError, match="future"):
        waiting_mean([0.3], 0.2, EXP, 0.25)


def test_common_reference_restores_poisson_prediction_ranking():
    rows = simulate_events(np.random.default_rng(17), POISSON, 2000)
    wrong = Process("poisson", rate=50)
    correct = landmark_losses(rows, POISSON)
    common = landmark_losses(rows, POISSON, wrong)
    old_self_scaled = landmark_losses(rows, wrong, wrong)
    assert np.sqrt(correct.mean()) == pytest.approx(1, abs=0.06)
    assert common.mean() > correct.mean()
    assert old_self_scaled.mean() < correct.mean()
    assert (
        gap_rmse(rows[:100], POISSON, wrong)["rmse"]
        > gap_rmse(rows[:100], POISSON, POISSON)["rmse"]
    )


def test_empty_paths_are_scored_censoring_not_dropped_or_zero_error():
    empty = [np.empty(0), np.empty(0)]
    losses = landmark_losses(empty, POISSON)
    assert (losses > 0).all()
    assert loss_test(losses, losses, 5, 19)["p_value"] == 1
    assert gap_rmse(empty, POISSON, POISSON) == dict(
        rmse=None, scored_paths=0, excluded_paths=2, completed_gaps=0
    )


def test_predictions_do_not_see_future_positions_and_no_path_mutation(monkeypatch):
    import scripts.report_rmse_validation as module

    seen = []
    original = module.waiting_mean

    def inspect(history, time, process, cap):
        assert np.all(history <= time)
        seen.append(time)
        return original(history, time, process, cap)

    monkeypatch.setattr(module, "waiting_mean", inspect)
    a, b = np.array([0.1, 0.2]), np.array([0.1, 0.21])
    saved = a.copy()
    landmark_losses([a, b], EXP)
    np.testing.assert_array_equal(a, saved)
    assert seen == [0, 0.25, 0.5, 0.75] * 2


def test_loss_swaps_equal_explicit_reassigned_group_rmse_statistic():
    x, y, n = np.array([1.0, 3.0, 8.0]), np.array([2.0, 9.0, 7.0]), 3
    seed, permutations = 17, 499
    result = loss_test(x, y, seed, permutations)
    generator = torch.Generator().manual_seed(seed)
    swaps = (
        torch.randint(2, (permutations, n), generator=generator).numpy().astype(bool)
    )
    expected = []
    for swap in swaps:
        xx, yy = np.where(swap, y, x), np.where(swap, x, y)
        expected.append((xx.mean() - yy.mean()) ** 2)
    np.testing.assert_allclose(result["null"], expected)
    assert result["observed"] == pytest.approx((x.mean() - y.mean()) ** 2)


def test_exact_replay_resume_and_refusal_of_tampered_source(tmp_path):
    source, output = tmp_path / "mmd", tmp_path / "rmse"
    run_cell("poisson_H0", 0, source, trials=2, n=4, permutations=19, grid=5)
    report = replay(source, output, descriptive=True)
    before = (output / "trials.jsonl").read_bytes()
    assert report["status"] == "completed"
    assert replay(source, output, descriptive=True) == report
    assert (output / "trials.jsonl").read_bytes() == before
    records = [json.loads(line) for line in before.splitlines()]
    assert all(
        row["prediction_rmse_on_x"]["reference"]
        == row["prediction_rmse_on_x"]["candidate"]
        for row in records
    )
    plan = json.loads((source / "plan.json").read_text())
    original = json.loads((source / "trials.jsonl").read_text().splitlines()[0])
    original["x_sha256"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        regenerate(plan, original)
    with pytest.raises(ValueError, match="Changed"):
        replay(source, output, descriptive=False)


def test_incomplete_campaign_and_incomplete_source_never_certified(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregate(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert len(summary["errors"]) == 20
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text(json.dumps(dict(trials=2)))
    (source / "trials.jsonl").write_text('{"trial":0}')
    with pytest.raises(ValueError, match="still being appended"):
        replay(source, tmp_path / "output")
