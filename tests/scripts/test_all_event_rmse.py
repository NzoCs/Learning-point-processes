"""All-event information, terminal censoring, numerical and replay checks."""

import json

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from scripts.all_event_rmse import all_event_losses, forecast_origins, forecasts
from scripts.report_experiment_validation import EXP, POISSON, Process, run_cell
from scripts.report_rmse_validation import (
    aggregate,
    calibration_row,
    loss_test,
    replay,
)


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def independent_means(rows, process, origins):
    result = []
    for time, prefix, path in zip(origins.times, origins.prefix, origins.path):
        history = rows[path][:prefix]
        if process.kind == "poisson":
            mu = process.rate
        else:
            mu = process.rate * (1 - process.branching)

        def intensity(s):
            ages = time + s - history
            if process.kind == "poisson":
                return mu
            if process.kind == "exp":
                return (
                    mu
                    + process.branching
                    * process.beta
                    * np.exp(-process.beta * ages).sum()
                )
            return (
                mu
                + process.branching
                * process.beta
                * (process.exponent - 1)
                * ((1 + process.beta * ages) ** (-process.exponent)).sum()
            )

        value = quad(
            lambda u: np.exp(-quad(intensity, 0, u, epsabs=1e-11)[0]),
            0,
            1 - time,
            epsabs=1e-11,
        )[0]
        result.append(value)
    return np.asarray(result)


def test_every_interval_and_terminal_censor_are_counted_including_empty_paths():
    rows = [np.empty(0), np.array([0.2, 0.7]), np.array([1.0])]
    origins = forecast_origins(rows)
    np.testing.assert_array_equal(origins.count, [1, 3, 1])
    np.testing.assert_allclose(origins.targets[origins.primary], [1, 0.2, 0.5, 0.3, 1])
    result = all_event_losses(rows, POISSON)
    assert len(result["path_mse"]) == 3
    assert result["path_mse"][0] > 0
    assert loss_test(result["path_mse"], result["path_mse"], 7, 19)["p_value"] == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        forecast_origins([np.array([0.2, 0.2])])
    boundary = forecast_origins([np.array([0.0, 0.5])])
    np.testing.assert_array_equal(boundary.prefix[boundary.primary], [0, 1, 2])
    np.testing.assert_allclose(boundary.targets[boundary.primary], [0, 0.5, 0.5])


@pytest.mark.parametrize(
    "process",
    [POISSON, EXP, Process("power", exponent=1.5), Process("power", exponent=3)],
)
def test_predictions_match_nested_independent_intensity_integrals(process):
    rows = [np.array([0.01, 0.04, 0.4, 0.999999]), np.empty(0)]
    origins = forecast_origins(rows)
    values = forecasts(rows, process, origins, chunk=3)
    expected = independent_means(rows, process, origins)
    np.testing.assert_allclose(values, expected, rtol=1e-8, atol=1e-11)
    losses = all_event_losses(rows, process, chunk=2, verify=True)
    squared = ((origins.targets - expected) / expected) ** 2
    for i in range(len(rows)):
        use = origins.primary & (origins.path == i)
        assert losses["path_mse"][i] == pytest.approx(squared[use].mean(), rel=1e-8)


def test_no_future_leakage_and_chunking_does_not_change_scores():
    a, b = [np.array([0.1, 0.4, 0.6])], [np.array([0.1, 0.4, 0.9])]
    oa, ob = forecast_origins(a), forecast_origins(b)
    va, vb = forecasts(a, EXP, oa), forecasts(b, EXP, ob)
    # Origins 0, .1 and .4 have the same visible past despite different futures.
    np.testing.assert_array_equal(va[:3], vb[:3])
    np.testing.assert_allclose(
        all_event_losses(a, EXP, chunk=1)["path_mse"],
        all_event_losses(a, EXP, chunk=256)["path_mse"],
        rtol=1e-12,
    )


def test_no_arbitrary_fixed_cap_and_final_censor_prediction_is_restricted():
    rows = [np.array([0.9])]
    process = Process("poisson", rate=1)
    origins = forecast_origins(rows)
    values = forecasts(rows, process, origins)
    assert values[0] == pytest.approx(1 - np.exp(-1))
    assert values[1] == pytest.approx(1 - np.exp(-0.1))
    assert values[0] > 0.25
    np.testing.assert_allclose(origins.targets[origins.primary], [0.9, 0.1])


def test_original_landmark_archive_and_full_replay_are_both_preserved(tmp_path):
    source, old, new = tmp_path / "mmd", tmp_path / "four", tmp_path / "all"
    run_cell("poisson_H0", 0, source, trials=2, n=4, permutations=19, grid=5)
    replay(source, old)
    before = (old / "trials.jsonl").read_bytes()
    report = replay(source, new, mode="all_events", landmark_source=old)
    assert report["status"] == "completed"
    rows = [
        json.loads(line) for line in (new / "trials.jsonl").read_text().splitlines()
    ]
    originals = [json.loads(line) for line in before.splitlines()]
    for a, b in zip(rows, originals):
        assert a["controls"]["landmark_original"] == b["rmse"]
        assert a["mmd"] == b["mmd"]
        assert min(a["origins_per_path"]) > 4
        assert a["origins_per_path"] == [n + 1 for n in a["events_per_path"]]
        assert len(a["rmse"]["path_mse_x"]) == 4  # Trajectories, not event gaps.
    saved = (new / "trials.jsonl").read_bytes()
    assert replay(source, new, mode="all_events", landmark_source=old) == report
    assert (new / "trials.jsonl").read_bytes() == saved
    assert (old / "trials.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="Changed"):
        replay(source, new, landmark_source=old)
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregate(tmp_path / "missing", mode="all_events")


def test_calibration_includes_new_controls_in_the_multiplicity_family():
    from scipy.stats import beta, binomtest

    records = [dict(rmse=dict(p_value=0.01 if i < 50 else 0.5)) for i in range(1000)]
    result = calibration_row("poisson_H0", "pooled", records, family_size=72)
    assert result["excess_p"] == binomtest(50, 1000, 0.05, alternative="greater").pvalue
    assert result["simultaneous_upper_99_over_24"] == beta.ppf(1 - 0.01 / 24, 51, 950)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_scores_match_without_fallback():
    rows = [np.array([0.01, 0.02, 0.4, 0.99]), np.empty(0)]
    for process in (EXP, Process("power", exponent=1.5)):
        a = all_event_losses(rows, process, device="cpu")
        b = all_event_losses(rows, process, device="cuda")
        for name in a:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-8, atol=1e-10)
