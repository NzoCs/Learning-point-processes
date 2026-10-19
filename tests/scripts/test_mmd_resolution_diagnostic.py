import json

import numpy as np
import pytest
import torch

from scripts import mmd_resolution_diagnostic as diagnostic


def test_event_path_retains_every_jump_time_and_empty_censoring():
    paths = diagnostic.event_paths([np.array([0.2, 0.21]), np.array([])], 2)
    np.testing.assert_array_equal(
        paths[0], [[0, 0], [0, 0.2], [0.5, 0.2], [0.5, 0.21], [1, 0.21], [1, 1]]
    )
    np.testing.assert_array_equal(
        paths[1], [[0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    )


def test_grid_can_erase_a_real_time_difference():
    a, b = [np.array([0.203, 0.204])], [np.array([0.209, 0.210])]
    torch.testing.assert_close(
        diagnostic.prepare_paths(a, 1, 100),
        diagnostic.prepare_paths(b, 1, 100),
        rtol=0,
        atol=0,
    )
    assert not torch.equal(diagnostic.event_paths(a, 1), diagnostic.event_paths(b, 1))
    assert not torch.equal(
        diagnostic.prepare_paths(a, 1, 400), diagnostic.prepare_paths(b, 1, 400)
    )


@pytest.mark.parametrize("row", [[0.2, 0.1], [0], [1], [np.inf], [-0.1]])
def test_invalid_events_fail_instead_of_being_clipped(row):
    with pytest.raises(ValueError):
        diagnostic.event_paths([np.array(row)], 1)


@pytest.mark.parametrize("kernel", diagnostic.KERNELS)
def test_endpoint_padding_does_not_change_signature(kernel):
    rows = [np.array([0.2]), np.array([])]
    paths = diagnostic.event_paths(rows, 2)
    padded = torch.cat([paths, paths[:, -1:, :].expand(-1, 7, -1)], dim=1)
    expected = diagnostic.kernel_gram(paths, kernel)
    actual = diagnostic.kernel_gram(padded, kernel)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    # A longer unrelated trajectory changes padding only, not these kernels.
    enlarged = diagnostic.event_paths([*rows, np.array([0.1, 0.2, 0.3])], 2)
    torch.testing.assert_close(
        diagnostic.kernel_gram(enlarged, kernel)[:2, :2],
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_repeated_seed_and_disjoint_selection_calibration():
    x, y, seed = diagnostic.independent_pair("poisson_H0", 0, "selection", n=4)
    a, b, again = diagnostic.independent_pair("poisson_H0", 0, "selection", n=4)
    c, _, other = diagnostic.independent_pair("poisson_H0", 0, "calibration", n=4)
    assert diagnostic.event_hash(x) == diagnostic.event_hash(a)
    assert diagnostic.event_hash(y) == diagnostic.event_hash(b)
    assert seed == again
    assert diagnostic.event_hash(x) != diagnostic.event_hash(c)
    assert seed != other


def test_baseline_and_event_tests_reproducible():
    x, y, seed = diagnostic.independent_pair("poisson_H0", 0, "selection", n=3)
    methods = ("grid100_linear", "events_rbf1")
    a = diagnostic.evaluate(x, y, 10, seed, methods, "cpu")
    b = diagnostic.evaluate(x, y, 10, seed, methods, "cpu")
    for m in methods:
        assert a[m]["p_value"] == b[m]["p_value"]
        assert a[m]["null"] == b[m]["null"]


def test_selection_uses_only_independent_selection_trials(tmp_path):
    for case in diagnostic.CASES:
        path = tmp_path / "selection" / case / "trials.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w") as handle:
            for i in range(32):
                handle.write(
                    json.dumps(
                        dict(
                            trial=i,
                            methods={
                                m: dict(p_value=0.01 if m.endswith("rbf3") else 0.5)
                                for m in diagnostic.CANDIDATES
                            },
                        )
                    )
                    + "\n"
                )
    diagnostic.select(tmp_path)
    selected = json.loads((tmp_path / "selection.json").read_text())
    assert selected["selected"] == {rep: f"{rep}_rbf3" for rep in diagnostic.REPS}
    # Later evaluated results cannot enter the selector, and freeze forbids reselection.
    (tmp_path / "summary.json").write_text('{"preferred":"linear"}')
    diagnostic.select(tmp_path)
    assert json.loads((tmp_path / "selection.json").read_text()) == selected


def test_freeze_refuses_changed_protocol_and_handles_json_sequences(tmp_path):
    path = tmp_path / "plan.json"
    diagnostic.freeze(path, dict(candidates=diagnostic.CANDIDATES))
    diagnostic.freeze(path, dict(candidates=diagnostic.CANDIDATES))
    with pytest.raises(ValueError, match="changed contract"):
        diagnostic.freeze(path, dict(candidates=[]))


def test_torn_record_and_wrong_p_value_are_rejected(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text('{"trial":0}')
    with pytest.raises(ValueError, match="Torn"):
        diagnostic.records(path)
    path.write_text(
        json.dumps(
            dict(trial=0, methods=dict(m=dict(p_value=0, observed=1, null=[0, 2])))
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="P-value"):
        diagnostic.records(path)
