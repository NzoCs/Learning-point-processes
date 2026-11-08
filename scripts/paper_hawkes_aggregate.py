"""Independent replay of finite-start Hawkes pilot outputs; never pool partial cells."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from scripts.aggregate_trained_test_pilot import pool_digest


def run(root, plan):
    rates, pairs = [], []
    for family, p in enumerate(plan["p_values"]):
        folder = root / f"evaluation-{family}"
        summary = json.loads((folder / "summary.json").read_text())
        if summary["status"] != "complete_finite_start_pilot":
            raise ValueError(f"Incomplete family {family}: {summary}")
        rows = [
            json.loads(s) for s in (folder / "trials.jsonl").read_text().splitlines()
        ]
        if len(rows) != plan["evaluation_trials"] or [r["trial"] for r in rows] != list(
            range(len(rows))
        ):
            raise ValueError("Missing or duplicated trials")
        rejected = []
        for row in rows:
            if (
                row["n"] != 128
                or row["burn"] != plan["burns"][-1]
                or len(row["tests"]) != 10
            ):
                raise ValueError("Observation/method count mismatch")
            for group, key in enumerate(["observed_sha256", "simulated_sha256"]):
                if (
                    pool_digest(folder / f"trial-{row['trial']:03d}-g{group}.npz")
                    != row[key]
                ):
                    raise ValueError("Input hash mismatch")
            flags = []
            for item in row["tests"]:
                null = np.asarray(item["permuted_statistics"])
                stat = item["observed_statistic"]
                if (
                    len(null) != plan["permutations"]
                    or not np.isfinite(null).all()
                    or not np.isfinite(stat)
                ):
                    raise ValueError("Invalid null vector")
                recomputed = (
                    1
                    + np.count_nonzero(
                        (null >= stat) | np.isclose(null, stat, rtol=1e-10, atol=1e-12)
                    )
                ) / (len(null) + 1)
                if abs(item["p_value"] - recomputed) > 1e-14:
                    raise ValueError("P-value mismatch")
                flags.append(recomputed <= 0.05)
            rejected.append(flags)
        decisions = np.asarray(rejected)
        for method, values in enumerate(decisions.T):
            k, n = int(values.sum()), len(values)
            ci = binomtest(k, n).proportion_ci(0.95, method="exact")
            simultaneous = binomtest(k, n).proportion_ci(1 - 0.01 / 10, method="exact")
            rates.append(
                dict(
                    family=family,
                    p=p,
                    method=method,
                    n=n,
                    rejections=k,
                    rate=k / n,
                    exact95_low=ci.low,
                    exact95_high=ci.high,
                    simultaneous99_low=simultaneous.low if p is None else None,
                    simultaneous99_high=simultaneous.high if p is None else None,
                    h0_excess_p=(
                        float(binomtest(k, n, p=0.05, alternative="greater").pvalue)
                        if p is None
                        else None
                    ),
                )
            )
        if p is not None:
            for i, j in combinations(range(10), 2):
                a, b = decisions[:, i], decisions[:, j]
                ab, ba = int((a & ~b).sum()), int((~a & b).sum())
                pairs.append(
                    dict(
                        family=family,
                        left=i,
                        right=j,
                        left_only=ab,
                        right_only=ba,
                        p=float(binomtest(ab, ab + ba).pvalue) if ab + ba else 1.0,
                    )
                )
    if len(pairs) != plan["power_pairwise_family"]:
        raise ValueError("Multiple-comparison family mismatch")
    previous = 0.0
    for rank, index in enumerate(
        sorted(range(len(pairs)), key=lambda i: pairs[i]["p"])
    ):
        previous = max(previous, min(1.0, pairs[index]["p"] * (len(pairs) - rank)))
        pairs[index]["holm_p"] = previous
    return dict(
        status="complete_finite_start_pilot",
        rates=rates,
        paired_comparisons=pairs,
        recomputed_p_values=len(plan["p_values"]) * plan["evaluation_trials"] * 10,
        excesses=[r for r in rates if r["p"] is None and r["h0_excess_p"] < 0.01 / 10],
        limitations="Finite-start B=500, T=1. 100 independent trials/cell is a pilot. Common exponential empty-past working predictor for loss; KSD not applied to hidden-burn targets. Read all burn/cost diagnostics; no stationarity or universal power claim.",
    )


def figures(root, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [
        "grid25",
        "grid50",
        "grid100",
        "grid200",
        "grid500",
        "events",
        "intervals",
        "count only",
        "linear grid100",
        "relative loss",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, name in enumerate(names):
        rows = [
            r for r in report["rates"] if r["p"] is not None and r["method"] == method
        ]
        x = [r["p"] for r in rows]
        y = np.array([r["rate"] for r in rows])
        ax.errorbar(
            x,
            y,
            yerr=[
                y - np.array([r["exact95_low"] for r in rows]),
                np.array([r["exact95_high"] for r in rows]) - y,
            ],
            marker=".",
            label=name,
            linewidth=1,
        )
    ax.set(
        xscale="log",
        xlabel="Convergent-kernel p",
        ylabel="Rejection rate (pilot; exact 95% intervals)",
        ylim=(0, 1.02),
        title="Finite-start Hawkes, B=500, T=1, n=128, 100 independent trials",
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "pilot-power.pdf")
    fig.savefig(root / "pilot-power.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    try:
        report = run(a.root, json.loads(a.plan.read_text()))
    except Exception as error:
        report = dict(
            status="incomplete_or_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    with (a.root / "analysis.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    if report["status"] == "complete_finite_start_pilot":
        with (a.root / "rates.csv").open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report["rates"][0]))
            w.writeheader()
            w.writerows(report["rates"])
        figures(a.root, report)
    print(json.dumps(report))
    raise SystemExit(
        2
        if report.get("excesses")
        else 0 if report["status"] == "complete_finite_start_pilot" else 1
    )
