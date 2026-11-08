import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.aggregate_trained_test_pilot import aggregate, pool_digest
from scripts.ruche_trained_test_pilot import configs, model_for_rates


def test_oracle_mark_alternative_preserves_total_rate_and_library_exposure(tmp_path):
    plan = json.loads(
        Path("configs/experiments/ruche_trained_pilot_20261108.json").read_text()
    )
    spec = plan["oracle_cases"][1]
    assert sum(spec["rates"]) == sum(spec["alternative_rates"])
    assert spec["rates"] != spec["alternative_rates"]
    model = model_for_rates(spec["rates"], tmp_path, "cpu")
    torch.manual_seed(17)
    batch = model.simulate_until(16, 1.0)
    assert torch.all(
        (batch.time_seqs[batch.valid_event_mask] > 0)
        & (batch.time_seqs[batch.valid_event_mask] < 1)
    )
    torch.testing.assert_close(model.rates, torch.tensor(spec["rates"]))
    settings = configs(plan, 3)
    assert len(settings) == 6 and all(
        c.horizon == 1 and c.num_event_types == 3 for c in settings
    )
    assert plan["statistics"]["calibration_excess_family"] == 2 * len(settings)
    assert (
        plan["statistics"]["power_pairwise_family"]
        == 2 * len(settings) * (len(settings) - 1) // 2
    )


def test_aggregate_replays_all_pvalues_and_rejects_changed_inputs(tmp_path):
    plan = json.loads(
        Path("configs/experiments/ruche_trained_pilot_20261108.json").read_text()
    )
    plan["trials_per_hypothesis"] = 1
    for case in range(2):
        folder = tmp_path / f"oracle-{case}"
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps(dict(status="complete")))
        for label in ["h0", "h1"]:
            for stream in range(2):
                path = folder / f"{label}-000-{stream}.npz"
                np.savez(
                    path,
                    times=np.zeros((128, 1)),
                    marks=np.zeros((128, 1), dtype=np.int64),
                    mask=np.zeros((128, 1), dtype=bool),
                )
            row = dict(
                n=128,
                trial=0,
                observed_sha256=pool_digest(folder / f"{label}-000-0.npz"),
                simulated_sha256=pool_digest(folder / f"{label}-000-1.npz"),
                tests=[
                    dict(
                        config=m,
                        observed_statistic=0.0,
                        permuted_statistics=[0.0] * m["n_samples"],
                        p_value=1.0,
                    )
                    for m in plan["methods"]
                ],
            )
            (folder / (label + "-trials.jsonl")).write_text(json.dumps(row) + "\n")
    report = aggregate(tmp_path, plan)
    assert report["recomputed_p_values"] == 24 and not report["excesses"]
    assert len(report["comparisons"]) == 30
    assert all(c["holm_p"] == 1 for c in report["comparisons"])
    np.savez(
        tmp_path / "oracle-0/h0-000-0.npz",
        times=np.ones((128, 1)),
        marks=np.zeros((128, 1), dtype=np.int64),
        mask=np.zeros((128, 1), dtype=bool),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        aggregate(tmp_path, plan)
