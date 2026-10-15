"""The historical check must catch changed results and silently dropped trials."""

import json

import pytest

from scripts.pipeline_calibration_regression import draw_hash, verify


def files(tmp_path):
    settings = dict(
        model="Hawkes",
        scenario="quota6",
        seed=1,
        event_quota=6,
        batch_sizes=[2],
        history_events=2,
        marks=2,
        H0_trials=2,
        H1_trials=0,
        cases=["H0"],
        kernel_config=dict(n_samples=99),
        protocol="same",
        observation="same",
        h0_family_size=36,
        h0_excess_family_threshold=0.01 / 36,
        fit=dict(checkpoint="fixed"),
    )
    first = dict(seed=41, observed=0.1, null=[0.2, 0.3], null_seed=100)
    second = dict(seed=42, observed=0.2, null=[0.3, 0.4], null_seed=101)
    baseline = dict(
        report=dict(settings, errors=[dict(case="H0", seed=42)]),
        successful_draw_sha256=dict(H0={"41": draw_hash(first)}),
    )
    after = dict(
        settings,
        status="completed",
        errors=[],
        model_frozen_verified=True,
        no_h0_excess_detected=True,
        results=[
            dict(hypothesis="H0", failed_trials=0, completed_trials=2, planned_trials=2)
        ],
        code=dict(commit="repair"),
    )
    output = tmp_path / "out"
    output.mkdir()
    before = tmp_path / "baseline.json"
    before.write_text(json.dumps(baseline))
    (output / "report.json").write_text(json.dumps(after))
    (output / "H0_draws.json").write_text(json.dumps([first, second]))
    return before, output, after, [first, second]


def test_exact_history_and_repaired_trial_are_both_accounted_for(tmp_path):
    before, output, _, _ = files(tmp_path)
    result = verify(before, output)
    assert result["previous_successful_draws_exact"] == 1
    assert result["original_failures_repaired"] == 1
    assert result["total_trials"] == 2


@pytest.mark.parametrize(
    "change",
    [
        "null",
        "null_rng",
        "missing_failed",
        "checkpoint",
        "protocol",
        "filtered",
        "missing_case",
    ],
)
def test_regression_does_not_allow_changed_science_or_filtered_failures(
    tmp_path, change
):
    before, output, report, draws = files(tmp_path)
    if change == "null":
        draws[0]["null"][0] += 1e-15
    elif change == "null_rng":
        draws[0]["null_seed"] += 1
    elif change == "missing_failed":
        draws[1]["seed"] = 99
    elif change == "checkpoint":
        report["fit"]["checkpoint"] = "changed"
    elif change == "protocol":
        report["marks"] = 3
    elif change == "filtered":
        report["errors"] = [dict(seed=42)]
    else:
        report["results"] = []
    (output / "report.json").write_text(json.dumps(report))
    (output / "H0_draws.json").write_text(json.dumps(draws))
    with pytest.raises(AssertionError):
        verify(before, output)
