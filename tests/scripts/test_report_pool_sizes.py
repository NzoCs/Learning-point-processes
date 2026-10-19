import json

import numpy as np
import pytest
import torch

from scripts import report_pool_sizes as pools
from scripts.report_experiment_validation import prepare_paths
from scripts.report_rmse_validation import loss_test


def test_provenance_can_move_between_cpu_and_gpu_nodes(monkeypatch):
    loaded = [False]

    class Kernel:
        def __init__(self, *args, **kwargs):
            pass

        def metadata(self):
            return dict(cuda_plugin_loaded=loaded[0], version="4", dyadic_order=0)

    monkeypatch.setattr(pools, "PySigLibKernel", Kernel)
    monkeypatch.setattr(
        pools.subprocess,
        "check_output",
        lambda args, **kw: "" if "status" in args else "frozen-commit",
    )
    cpu = pools.provenance()
    loaded[0] = True
    gpu = pools.provenance()
    assert cpu == gpu
    assert "cuda_plugin_loaded" not in cpu["kernel"]


def test_prefix_indices_keep_x_and_y_groups_separate():
    assert pools.prefix_indices(1024, 128, "cpu").tolist() == [
        *range(128),
        *range(1024, 1152),
    ]
    for n in (1, 1025):
        with pytest.raises(ValueError):
            pools.prefix_indices(1024, n, "cpu")


def test_same_pair_repeats_and_pilot_evaluation_have_separate_namespaces():
    a, b, s = pools.pair("poisson_H0", 0, 0, "evaluation", n=4)
    x, y, t = pools.pair("poisson_H0", 0, 0, "evaluation", n=4)
    p, _, u = pools.pair("poisson_H0", 0, 0, "pilot", n=4)
    assert (pools.event_hash(a), pools.event_hash(b), s) == (
        pools.event_hash(x),
        pools.event_hash(y),
        t,
    )
    assert pools.event_hash(a) != pools.event_hash(p) and s != u
    assert pools.entropy("poisson_H0", 0, 0, 1, "evaluation")[0] not in (
        2026101931,
        2026102037,
        2026101601,
    )


def test_timing_control_preserves_counts_even_for_maximum_nested_bank():
    x, y, _ = pools.pair("poisson_timing_H0", 1, 0, "evaluation", n=1024)
    assert [len(v) for v in x] == [len(v) for v in y]
    assert pools.event_hash(x) != pools.event_hash(y)


def test_nesting_shares_random_swaps_and_matches_legacy_at_full_size():
    seed = 517
    swap = pools.signs(seed, 6)
    loss = np.array([1.0, 4.0, 2.0, 3.0, 7.0, 2.0])
    other = np.array([2.0, 3.0, 5.0, 1.0, 8.0, 4.0])
    got = pools.stats((swap @ torch.from_numpy(loss - other) / 6).square())
    legacy = loss_test(loss, other, seed, 499)
    assert all(got[k] == legacy[k] for k in ("observed", "null", "p_value"))
    # A nested test uses the very same signs for MMD and RMSE, not a reshaped RNG stream.
    assert torch.equal(swap[:, :2], pools.signs(seed, 6)[:, :2])


def test_nested_gram_matches_separate_kernel_and_explicit_paired_statistic():
    x = [np.array([0.1, 0.2]), np.array([0.3]), np.array([]), np.array([0.2, 0.7])]
    y = [np.array([0.4]), np.array([0.2, 0.8]), np.array([0.6]), np.array([0.1])]
    reference = pools.Process("poisson", rate=2)
    got = pools.evaluate(x, y, reference, 1.5, 719, "cpu", sizes=(2, 4))
    paths = prepare_paths(x[:2] + y[:2], 1.5, 100)
    with torch.no_grad():
        gram = pools.PySigLibKernel(pools.RBFKernel(0.3), 0, max_batch=16).compute_Gram(
            paths, paths
        )
    np.testing.assert_allclose(
        got["sizes"]["2"]["mmd"]["observed"],
        float(
            (
                gram[0, 1]
                + gram[2, 3]
                - gram[0, 3]
                - gram[1, 2]
                + gram[1, 0]
                + gram[3, 2]
                - gram[1, 2]
                - gram[0, 3]
            )
            / 2
        ),
        rtol=1e-8,
        atol=1e-8,
    )
    expected = pools.stats(pools.swap_statistics(gram, pools.signs(719, 4)[:, :2]))
    assert got["sizes"]["2"]["mmd"]["p_value"] == expected["p_value"]
    np.testing.assert_allclose(
        got["sizes"]["2"]["mmd"]["null"], expected["null"], rtol=1e-8, atol=1e-8
    )
    assert got["sizes"]["2"]["x_sha256"] == pools.event_hash(x[:2])
    assert got["origins_per_path"] == [len(p) + 1 for p in x + y]
    standalone = pools.all_event_losses(x[:2] + y[:2], reference)
    np.testing.assert_allclose(
        standalone["path_mse"], got["path_mse_x"][:2] + got["path_mse_y"][:2]
    )


def test_evaluation_rejects_unnested_or_duplicate_sizes():
    for sizes in ((4, 2), (2, 2), (2, 3)):
        with pytest.raises(ValueError):
            pools.evaluate(
                [np.array([])] * 4,
                [np.array([])] * 4,
                pools.Process("poisson"),
                1,
                1,
                "cpu",
                sizes=sizes,
            )


def test_frozen_plan_cannot_be_replaced(tmp_path):
    path = tmp_path / "plan.json"
    pools.freeze(path, dict(sizes=pools.SIZES))
    pools.freeze(path, dict(sizes=pools.SIZES))
    with pytest.raises(ValueError, match="Changed frozen"):
        pools.freeze(path, dict(sizes=[128]))


def test_no_silent_torn_record_or_wrong_trial_resume(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text('{"trial":200}')
    with pytest.raises(ValueError, match="Torn"):
        pools.read_records(path, 200)
    path.write_text(json.dumps(dict(trial=201)) + "\n")
    with pytest.raises(ValueError, match="Noncontiguous"):
        pools.read_records(path, 200)
    path.write_text(json.dumps(dict(trial=200)) + "\n")
    assert pools.read_records(path, 200) == [dict(trial=200)]


def test_incomplete_campaign_refuses_a_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(pools, "contract", lambda root: {})
    with pytest.raises(FileNotFoundError):
        pools.aggregate(tmp_path)
    assert not (tmp_path / "summary.json").exists()
