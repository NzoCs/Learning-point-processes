"""Paired experiment: same raw data and permutations for both observation rules.

The experimental truncated statistic is recomputed from raw data after every
permutation. Production defaults and historical references remain unchanged.
"""

import argparse
import hashlib
import json
import time
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
from scripts.scientific_calibration import (
    gram_statistic,
    sample_process,
    wilson_interval,
)


def count_ge(observed, null):
    # Conservative treatment of theoretical ties affected by floating summation.
    return int(
        (
            (null >= observed) | torch.isclose(null, observed, rtol=1e-10, atol=1e-12)
        ).sum()
    )


def permutation_p_value(observed, null):
    # Divide integer counts in Python float64, so 1/20 is exactly the .05 threshold.
    return (1 + count_ge(observed, null)) / (1 + len(null))


class RecomputedTruncationTest(MMDTwoSampleTest):
    """Experimental permutation test, never permute already truncated samples."""

    def _permutation_test(self, batch_x, batch_y):
        if len(batch_x.time_seqs) != len(batch_y.time_seqs):
            raise ValueError("Pairwise truncation requires equal group sizes")
        x, y = self._truncate_to_min_time(batch_x, batch_y)
        observed = self.mmd(x, y)
        pooled = self._concat_batches(batch_x, batch_y)
        n = len(batch_x.time_seqs)
        null = []
        for _ in range(self.n_samples):
            order = torch.randperm(2 * n)
            x, y = self._truncate_to_min_time(
                self._select_batch(pooled, order[:n]),
                self._select_batch(pooled, order[n:]),
            )
            null.append(self.mmd(x, y))
        return observed, torch.stack(null)

    def compute_statistics(self, batch_x, batch_y, simulations=None, accumulate=True):
        if simulations is not None:
            raise ValueError("This experiment validates permutations only")
        observed, null = self._permutation_test(batch_x, batch_y)
        if accumulate:
            self._accumulate(observed, null)
        return {
            "observed_statistic": observed,
            "permuted_statistics": null,
            "p_value": torch.tensor(
                permutation_p_value(observed, null), dtype=torch.float64
            ),
            "num_null_samples": len(null),
            "null_method": "permutation_recomputed_truncation",
        }


def paired_statistics(test, x, y, orders):
    """Efficient full-pool Grams; equivalence to actual MMD is tested separately."""
    n = len(x.time_seqs)
    if n != len(y.time_seqs):
        raise ValueError("Pairwise truncation requires equal group sizes")
    pool = test._concat_batches(x, y)
    gram = test.kernel.compute_gram_matrix(pool, pool)
    direct, truncated = [], []
    # The identity represents the observation; every other order is a null draw.
    indices = torch.arange(2 * n)
    for order in orders:
        direct.append(gram_statistic(gram, order[:n], order[n:]))
        px, py = test._truncate_to_min_time(
            test._select_batch(pool, order[:n]), test._select_batch(pool, order[n:])
        )
        prepared_pool = test._concat_batches(px, py)
        prepared_gram = test.kernel.compute_gram_matrix(prepared_pool, prepared_pool)
        truncated.append(gram_statistic(prepared_gram, indices[:n], indices[n:]))
    return {
        "without_truncation": torch.stack(direct),
        "recomputed_pairwise_truncation": torch.stack(truncated),
    }


def run_comparison(
    trials=500, power_trials=200, permutations=99, seed=20261012, output=None
):
    if min(trials, power_trials, permutations) < 1:
        raise ValueError("Trial and permutation counts must be positive")
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    orders_rng = torch.Generator().manual_seed(seed)
    test = MMDTwoSampleTest(
        SIGKernel(RBFKernel(), "counting_grid", 8, 0, 1), permutations
    )
    results, started = [], time.monotonic()
    for process in ("poisson", "hawkes"):
        for rate_y in (2.0, 4.0, 8.0):
            count = trials if rate_y == 2.0 else power_trials
            values = {
                name: []
                for name in ("without_truncation", "recomputed_pairwise_truncation")
            }
            raw_digest, order_digest = hashlib.sha256(), hashlib.sha256()
            for trial in range(count):
                x, y = sample_process(rng, process), sample_process(
                    rng, process, rate=rate_y
                )
                orders = torch.stack(
                    [torch.arange(16)]
                    + [
                        torch.randperm(16, generator=orders_rng)
                        for _ in range(permutations)
                    ]
                )
                raw_digest.update(
                    x.time_seqs.numpy().tobytes() + y.time_seqs.numpy().tobytes()
                )
                order_digest.update(orders.numpy().tobytes())
                statistics = paired_statistics(test, x, y, orders)
                for protocol, stats in statistics.items():
                    values[protocol].append(permutation_p_value(stats[0], stats[1:]))
                if (trial + 1) % 50 == 0:
                    print(
                        f"{process} rates 2/{rate_y:g}: {trial + 1}/{count}", flush=True
                    )
            decisions = {name: np.asarray(v) <= 0.05 for name, v in values.items()}
            a, b = decisions.values()
            difference = b.astype(float) - a.astype(float)
            delta_se = (
                float(difference.std(ddof=1) / np.sqrt(count)) if count > 1 else None
            )
            for protocol, p_values in values.items():
                rejected = int(decisions[protocol].sum())
                results.append(
                    {
                        "process": process,
                        "rate_x": 2.0,
                        "rate_y": rate_y,
                        "hypothesis": "H0" if rate_y == 2.0 else "H1",
                        "protocol": protocol,
                        "trials": count,
                        "rejections": rejected,
                        "rejection_rate": rejected / count,
                        "wilson_95": wilson_interval(rejected, count),
                        "p_values": p_values,
                        "raw_data_sha256": raw_digest.hexdigest(),
                        "permutations_sha256": order_digest.hexdigest(),
                    }
                )
            results[-1]["paired_comparison"] = {
                "delta_rejection_rate_truncated_minus_direct": float(difference.mean()),
                "delta_standard_error": delta_se,
                "only_truncated_rejects": int((~a & b).sum()),
                "only_direct_rejects": int((a & ~b).sum()),
            }
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps({"status": "in_progress", "results": results}, indent=2)
                    + "\n"
                )
    report = {
        "status": "completed",
        "seed": seed,
        "trials_H0": trials,
        "trials_H1": power_trials,
        "permutations": permutations,
        "samples_per_group": 8,
        "events_per_sequence": 5,
        "alpha": 0.05,
        "reject_rule": "p <= .05",
        "minimum_p": 1 / (permutations + 1),
        "numerical_ties": {
            "rtol": 1e-10,
            "atol": 1e-12,
            "rule": "count as greater-or-equal",
        },
        "p_value_arithmetic": "integer count divided in Python float64",
        "kernel": "pysiglib/RBF/counting_grid/grid8/dyadic0",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "duration_seconds": time.monotonic() - started,
        "scope": "Independent scalar fixed-count Poisson and Hawkes samples; same raw data and orders in both protocols; no fitted model, conditional simulation or batch aggregation certification",
        "results": results,
    }
    if output:
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--power-trials", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/truncation-comparison/report.json"),
    )
    args = parser.parse_args()
    report = run_comparison(
        args.trials, args.power_trials, args.permutations, args.seed, args.output
    )
    for row in report["results"]:
        print(
            row["process"],
            row["rate_y"],
            row["protocol"],
            row["rejection_rate"],
            row["wilson_95"],
            flush=True,
        )


if __name__ == "__main__":
    main()
