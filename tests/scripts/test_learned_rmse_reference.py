"""Likelihood, gradients, data separation and immutable matched replays."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from scripts.learned_rmse_reference import fit_reference, likelihood_data, nll_gradient
from scripts.report_experiment_validation import (
    Process,
    event_hash,
    run_cell,
    simulate_events,
)
from scripts.report_learned_rmse import (
    PROTOCOL,
    aggregate,
    calibration_entry,
    digest,
    frozen_records,
    load_fit,
    provenance,
    replay,
)
from scripts.report_rmse_validation import replay as known_replay


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", ["exp", "power"])
def test_training_likelihood_matches_independent_integral_and_gradient(kind):
    rows = [np.array([0.1, 0.13, 0.7]), np.array([]), np.array([0.4, 0.9])]
    theta = np.array(
        [np.log(2.0), 0.4, np.log(3.0)] + ([np.log(1.3)] if kind == "power" else [])
    )
    data = likelihood_data(rows)
    value, gradient = nll_gradient(theta, data, kind)
    mu, eta, beta = 2.0, 0.4, 3.0
    exponent = 2.3

    def kernel(age):
        return (
            eta
            * beta
            * (
                np.exp(-beta * age)
                if kind == "exp"
                else (exponent - 1) * (1 + beta * age) ** (-exponent)
            )
        )

    expected = 0.0
    for row in rows:
        expected += mu + sum(quad(kernel, 0, 1 - t, epsabs=1e-11)[0] for t in row)
        expected -= sum(
            np.log(mu + sum(kernel(t - u) for u in row[:i])) for i, t in enumerate(row)
        )
    assert value == pytest.approx(expected / len(rows), abs=1e-11)
    step = 1e-5
    numeric = np.asarray(
        [
            (
                nll_gradient(theta + step * e, data, kind)[0]
                - nll_gradient(theta - step * e, data, kind)[0]
            )
            / (2 * step)
            for e in np.eye(len(theta))
        ]
    )
    np.testing.assert_allclose(gradient, numeric, atol=1e-7, rtol=1e-6)


def test_poisson_fit_is_exact_count_over_all_exposure_including_empty_paths():
    rows = [np.array([]), np.array([0.2, 0.4]), np.array([0.7])]
    result = fit_reference(rows, "poisson")
    assert result["reference"]["rate"] == 1.0
    assert result["training_sha256"] == event_hash(rows)
    with pytest.raises(ValueError, match="No training events"):
        fit_reference([np.empty(0)], "poisson")
    with pytest.raises(ValueError, match="Strictly increasing"):
        fit_reference([np.array([0.2, 0.2])], "exp")


@pytest.mark.parametrize("kind", ["exp", "power"])
def test_fit_depends_only_on_training_and_improves_training_objective(kind):
    rows = simulate_events(
        np.random.default_rng(719),
        Process(kind, rate=12, branching=0.6, beta=8),
        64,
        burn=0,
    )
    a = fit_reference(rows, kind)
    b = fit_reference(rows, kind)
    assert a == b
    assert a["training_paths"] == 64
    selected = a["optimizer_runs"][a["selected_start"]]
    assert selected["success"]
    # The fitter never receives the generating parameters: its estimate need not
    # equal them, but it must improve on a data-derived Poisson reference.
    rate = sum(map(len, rows)) / len(rows)
    poisson_nll = rate - sum(map(len, rows)) * np.log(rate) / len(rows)
    assert selected["objective"] <= poisson_nll + 1e-6
    assert 0 <= a["reference"]["branching"] < 1


def training_fixture(root, monkeypatch):
    import scripts.report_learned_rmse as module

    current = provenance()
    current["tracked_dirty"] = False
    monkeypatch.setattr(module, "provenance", lambda: current)
    root.mkdir()
    rows = [np.array([0.1, 0.4, 0.8]), np.array([0.2])]
    bank = root / "poisson-0.npz"
    np.savez_compressed(bank, times=np.concatenate(rows), offsets=[0, 3, 4])
    fit = fit_reference(rows, "poisson")
    fit.update(
        protocol=PROTOCOL,
        provenance=current,
        cohort=0,
        family="poisson",
        bank_sha256=digest(bank),
    )
    f = root / "poisson-0.json"
    f.write_text(json.dumps(fit))
    (root / "training-complete.json").write_text(
        json.dumps(dict(status="completed", fits={f.name: digest(f)}))
    )
    return fit


def test_matched_replay_frozen_training_and_resume(tmp_path, monkeypatch):
    training = tmp_path / "training"
    fit = training_fixture(training, monkeypatch)
    mmd, old, known, new = [tmp_path / s for s in ("mmd", "old", "known", "learned")]
    run_cell("poisson_H0", 0, mmd, trials=2, n=4, permutations=19, grid=5)
    known_replay(mmd, old)
    known_replay(mmd, known, mode="all_events", landmark_source=old)
    before = {
        f: f.read_bytes()
        for d in (mmd, known, training)
        for f in d.iterdir()
        if f.is_file()
    }
    result = replay(known, training, new)
    assert result["status"] == "completed"
    saved = (new / "trials.jsonl").read_bytes()
    assert replay(known, training, new) == result
    assert (new / "trials.jsonl").read_bytes() == saved
    assert all(f.read_bytes() == raw for f, raw in before.items())
    records = frozen_records(new / "trials.jsonl")
    for a, b in zip(records, frozen_records(known / "trials.jsonl")):
        assert a["known_reference"] == b["rmse"]
        assert a["mmd"] == b["mmd"]
        assert len(a["rmse"]["path_mse_x"]) == 4
        assert a["origins_per_path"] == b["origins_per_path"]
    path = training / "poisson-0.json"
    changed = dict(fit, reference=asdict(Process("poisson", rate=50)))
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Frozen training source changed"):
        load_fit(training, "poisson", 0)


def test_partial_frozen_source_is_rejected_without_truncation(tmp_path):
    path = tmp_path / "trials.jsonl"
    raw = b'{"trial":0,"status":"completed"}\n{"trial":'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="Incomplete frozen archive"):
        frozen_records(path)
    assert path.read_bytes() == raw


def test_missing_cells_cannot_be_reported_as_a_complete_campaign(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregate(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "incomplete" and len(summary["errors"]) == 20


def test_trained_campaign_uses_prespecified_sixty_test_family():
    from scipy.stats import beta, binomtest

    ps = [0.01] * 70 + [0.5] * 930
    result = calibration_entry("poisson_H0", "pooled", "known_reference", ps)
    assert result["method"] == "known_reference"
    assert result["h0_excess_detected"] == (
        binomtest(70, 1000, 0.05, alternative="greater").pvalue < 0.01 / 60
    )
    assert result["simultaneous_upper_99_over_20"] == beta.ppf(1 - 0.01 / 20, 71, 930)
