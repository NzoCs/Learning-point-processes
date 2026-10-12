"""Seeded, opt-in calibration experiment; not a replacement for a validity proof.

Run: python -m scripts.scientific_calibration --trials 200 --permutations 99
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    MMDTwoSampleTest,
)
from new_ltpp.shared_types import Batch


def sample_process(rng, process, size=8, events=5, rate=2.0):
    """Independent scalar reference generators, fixed event count, no burn-in.

    Hawkes intensity = rate + sum .3 * exp(-(t-ti)); branching ratio .3.
    Implements scalar Ogata rejection; Poisson uses exact exponential gaps.
    """
    rows = []
    for _ in range(size):
        if process == "poisson":
            times = np.cumsum(rng.exponential(1 / rate, size=events))
        elif process == "hawkes":
            t, excitation, times = 0.0, 0.0, []
            while len(times) < events:
                bound = rate + excitation
                dt = rng.exponential(1 / bound)
                t += dt
                excitation *= math.exp(-dt)
                if rng.random() * bound <= rate + excitation:
                    times.append(t)
                    excitation += 0.3
            times = np.asarray(times)
        else:
            raise ValueError(f"Unknown reference process: {process}")
        rows.append(times)
    times = torch.tensor(np.asarray(rows), dtype=torch.float64)
    gaps = torch.diff(times, dim=1, prepend=torch.zeros(size, 1, dtype=times.dtype))
    return Batch(
        times,
        gaps,
        torch.zeros_like(times, dtype=torch.long),
        torch.ones_like(times, dtype=torch.bool),
    )


def gram_statistic(gram, x, y):
    """Unbiased statistic from a label-independent pooled Gram matrix."""
    xx, yy, xy = gram[x][:, x], gram[y][:, y], gram[x][:, y]
    n, m = len(x), len(y)
    if min(n, m) < 2:
        raise ValueError("At least two samples per group are required")
    return (
        (xx.sum() - xx.diag().sum()) / (n * (n - 1))
        + (yy.sum() - yy.diag().sum()) / (m * (m - 1))
        - 2 * xy.mean()
    )


def fast_permutation(test, x, y):
    """SIG pooled normalization is label invariant; tested against public API."""
    pooled = test._concat_batches(x, y)
    gram = test.kernel.compute_gram_matrix(pooled, pooled)
    n, total = len(x.time_seqs), len(pooled.time_seqs)
    ids = torch.arange(total)
    observed = gram_statistic(gram, ids[:n], ids[n:])
    null = []
    for _ in range(test.n_samples):
        ids = torch.randperm(total)
        null.append(gram_statistic(gram, ids[:n], ids[n:]))
    null = torch.stack(null)
    return float(((null >= observed).sum() + 1) / (len(null) + 1))


def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return [center - half, center + half]


def run_calibration(trials=200, permutations=99, seed=20261009):
    """Compare the current production observation rule to the raw reference."""
    from scripts.compare_truncation import run_comparison

    return run_comparison(
        trials=trials, power_trials=trials, permutations=permutations, seed=seed
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/scientific-calibration/report.json"),
    )
    args = parser.parse_args()
    report = run_calibration(args.trials, args.permutations, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["results"]:
        print(
            row["process"],
            row["hypothesis"],
            row["rate_y"],
            row["protocol"],
            row["rejection_rate"],
            row["wilson_95"],
        )


if __name__ == "__main__":
    main()
