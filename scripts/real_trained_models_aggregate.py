"""Audit each actual trained model; keep unsupported methods explicitly unavailable."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from scripts.aggregate_trained_test_pilot import pool_digest


def aggregate(root, plan):
    models = []
    comparisons = []
    for index in range(4):
        cell = root / "cells" / str(index)
        fit = json.loads((cell / "fit.json").read_text())
        entry = dict(index=index, fit=fit, methods=[], status="unavailable")
        if fit["status"] != "trained":
            models.append(entry)
            continue
        directory = cell / "benchmark"
        summary = json.loads((directory / "summary.json").read_text())
        entry["benchmark_summary"] = summary
        if summary["status"] != "complete_supported_substudies":
            models.append(entry)
            continue
        available = summary["availability"]
        active = [a["method"] for a in available if a["status"] == "supported"]
        decisions = {}
        for label in ["h0", "time_compression"]:
            rows = [
                json.loads(s)
                for s in (directory / (label + "-trials.jsonl"))
                .read_text()
                .splitlines()
            ]
            if len(rows) != plan["trials_per_hypothesis"] or [
                r["trial"] for r in rows
            ] != list(range(len(rows))):
                raise ValueError("Incomplete/duplicated trials, no exclusions")
            by_method = {i: [] for i in active}
            for row in rows:
                digests = [
                    pool_digest(directory / f'{label}-{row["trial"]:03d}-g{g}.npz')
                    for g in range(2)
                ]
                if [t["method_index"] for t in row["methods"]] != active:
                    raise ValueError("A supported method is missing")
                for record in row["methods"]:
                    if [
                        record["observed_sha256"],
                        record["simulated_sha256"],
                    ] != digests:
                        raise ValueError("Source data hash mismatch")
                    test = record["tests"][0]
                    null = np.asarray(test["permuted_statistics"])
                    stat = test["observed_statistic"]
                    if (
                        not np.isfinite(null).all()
                        or not np.isfinite(stat)
                        or len(null) != test["config"]["n_samples"]
                    ):
                        raise ValueError("Invalid statistics")
                    p = (
                        1
                        + np.count_nonzero(
                            (null >= stat)
                            | np.isclose(null, stat, rtol=1e-10, atol=1e-12)
                        )
                    ) / (1 + len(null))
                    if abs(p - test["p_value"]) > 1e-14:
                        raise ValueError("P-value mismatch")
                    by_method[record["method_index"]].append(p <= 0.05)
            decisions[label] = by_method
            for method, values in by_method.items():
                k, n = int(sum(values)), len(values)
                ci = binomtest(k, n).proportion_ci(0.95, method="exact")
                entry["methods"].append(
                    dict(
                        method=method,
                        hypothesis=label,
                        n=n,
                        rejections=k,
                        rate=k / n,
                        exact95=[ci.low, ci.high],
                        h0_excess_p=(
                            float(binomtest(k, n, p=0.05, alternative="greater").pvalue)
                            if label == "h0"
                            else None
                        ),
                    )
                )
        for i, j in combinations(range(len(plan["methods"])), 2):
            if i not in active or j not in active:
                comparisons.append(
                    dict(
                        index=index,
                        left=i,
                        right=j,
                        status="unavailable",
                        p=None,
                        holm_p=None,
                    )
                )
                continue
            a, b = np.asarray(decisions["time_compression"][i]), np.asarray(
                decisions["time_compression"][j]
            )
            ab, ba = int((a & ~b).sum()), int((~a & b).sum())
            comparisons.append(
                dict(
                    index=index,
                    left=i,
                    right=j,
                    status="computed",
                    left_only=ab,
                    right_only=ba,
                    p=float(binomtest(ab, ab + ba).pvalue) if ab + ba else 1.0,
                )
            )
        entry["status"] = (
            "complete_all_methods"
            if summary["full_suite_complete"]
            else "complete_supported_methods_only"
        )
        models.append(entry)
    actual = [i for i, c in enumerate(comparisons) if c["status"] == "computed"]
    previous = 0.0
    for rank, i in enumerate(sorted(actual, key=lambda i: comparisons[i]["p"])):
        previous = max(
            previous,
            min(1.0, comparisons[i]["p"] * (plan["paired_comparison_family"] - rank)),
        )
        comparisons[i]["holm_p"] = previous
    return dict(
        status=(
            "complete"
            if all(m["status"] == "complete_all_methods" for m in models)
            else "partial_with_explicit_blocked_substudies"
        ),
        models=models,
        paired_comparisons=comparisons,
        calibration_excess_threshold=0.01 / plan["calibration_family"],
        limitations=plan["scope"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    try:
        r = aggregate(a.root, json.loads(a.plan.read_text()))
    except Exception as e:
        r = dict(
            status="audit_failed_or_incomplete",
            error_type=type(e).__name__,
            error=str(e),
        )
    with (a.root / "analysis.json").open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
    print(json.dumps(r))
    raise SystemExit(0 if r["status"] == "complete" else 2)
