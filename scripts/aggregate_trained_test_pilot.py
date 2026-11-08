"""Verify every pilot p-value and report complete oracle cells only."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def pool_digest(path):
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in ["times", "marks", "mask"]:
            a = np.ascontiguousarray(data[key])
            digest.update(str((a.shape, a.dtype)).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def aggregate(root, plan):
    results, comparisons = [], []
    for case, spec in enumerate(plan["oracle_cases"]):
        folder = root / f"oracle-{case}"
        summary = json.loads((folder / "summary.json").read_text())
        if summary["status"] != "complete":
            raise ValueError(f"Oracle case {case} incomplete: {summary}")
        decisions = {}
        for label in ["h0", "h1"]:
            rows = [
                json.loads(line)
                for line in (folder / (label + "-trials.jsonl"))
                .read_text()
                .splitlines()
            ]
            if len(rows) != plan["trials_per_hypothesis"] or [
                r["trial"] for r in rows
            ] != list(range(len(rows))):
                raise ValueError("Incomplete or duplicated trials")
            rejection = []
            for row in rows:
                if row["n"] != 128 or len(row["tests"]) != len(plan["methods"]):
                    raise ValueError("Unexpected pool or test count")
                for stream in [0, 1]:
                    path = folder / f"{label}-{row['trial']:03d}-{stream}.npz"
                    key = "observed_sha256" if stream == 0 else "simulated_sha256"
                    if pool_digest(path) != row[key]:
                        raise ValueError("Source trajectory hash mismatch")
                rejected = []
                for test in row["tests"]:
                    null = np.asarray(test["permuted_statistics"])
                    observed = test["observed_statistic"]
                    if not np.isfinite(null).all() or not np.isfinite(observed):
                        raise ValueError("Nonfinite statistic")
                    if len(null) != test["config"]["n_samples"]:
                        raise ValueError("Null sample count mismatch")
                    p = (
                        1
                        + np.count_nonzero(
                            (null >= observed)
                            | np.isclose(null, observed, rtol=1e-10, atol=1e-12)
                        )
                    ) / (1 + len(null))
                    if abs(p - test["p_value"]) > 1e-14:
                        raise ValueError("P-value replay mismatch")
                    rejected.append(p <= plan["statistics"]["alpha"])
                rejection.append(rejected)
            decisions[label] = np.asarray(rejection)
            for method, values in enumerate(decisions[label].T):
                count, n = int(values.sum()), len(values)
                interval = binomtest(count, n).proportion_ci(
                    confidence_level=0.95, method="exact"
                )
                excess = float(
                    binomtest(count, n, p=0.05, alternative="greater").pvalue
                )
                results.append(
                    dict(
                        case=spec["name"],
                        hypothesis=label,
                        method=method,
                        rejections=count,
                        n=n,
                        rate=count / n,
                        exact95=[interval.low, interval.high],
                        h0_excess_p=excess if label == "h0" else None,
                    )
                )
        for left, right in combinations(range(len(plan["methods"])), 2):
            a, b = decisions["h1"][:, left], decisions["h1"][:, right]
            ab, ba = int((a & ~b).sum()), int((~a & b).sum())
            p = float(binomtest(ab, ab + ba, p=0.5).pvalue) if ab + ba else 1.0
            comparisons.append(
                dict(
                    case=spec["name"],
                    left=left,
                    right=right,
                    left_only=ab,
                    right_only=ba,
                    p=p,
                )
            )
    if len(comparisons) != plan["statistics"]["power_pairwise_family"]:
        raise ValueError("Comparison family mismatch")
    previous = 0.0
    for rank, index in enumerate(
        sorted(range(len(comparisons)), key=lambda i: comparisons[i]["p"])
    ):
        previous = max(
            previous, min(1.0, comparisons[index]["p"] * (len(comparisons) - rank))
        )
        comparisons[index]["holm_p"] = previous
    h0 = [r for r in results if r["hypothesis"] == "h0"]
    if len(h0) != plan["statistics"]["calibration_excess_family"]:
        raise ValueError("Calibration family mismatch")
    return dict(
        status="complete_oracle_pilot",
        results=results,
        comparisons=comparisons,
        excesses=[r for r in h0 if r["h0_excess_p"] < 0.01 / len(h0)],
        recomputed_p_values=2
        * len(plan["oracle_cases"])
        * plan["trials_per_hypothesis"]
        * len(plan["methods"]),
        limitations="Oracle Poisson pilot only, 100 trials/cell. Trained-model capability and real-data audit are reported separately; no general training or power claim.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    try:
        report = aggregate(a.root, json.loads(a.plan.read_text()))
    except Exception as error:
        report = dict(
            status="blocked_incomplete_or_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    with (a.root / "oracle-analysis.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report))
    raise SystemExit(
        2
        if report.get("excesses")
        else 0 if report["status"] == "complete_oracle_pilot" else 1
    )
