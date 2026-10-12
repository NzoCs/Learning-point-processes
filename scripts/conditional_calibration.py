"""Opt-in calibration of heterogeneous histories and product-group batch sums."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
    MMDTwoSampleTest,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test import (
    pair_swap_indices,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)
from scripts.compare_truncation import paired_statistics
from scripts.extended_scientific_validation import (
    batch_from_arrays,
    code_provenance,
    fingerprint,
    fit_model,
    marked_reference,
    select,
)
from scripts.scientific_calibration import gram_statistic, wilson_interval


def heterogeneous_histories(size):
    rows_t, rows_m = [], []
    for i in range(size):
        rows_t.append([0.1, 0.2, 0.3] if i % 2 == 0 else [0.2, 2.0, 5.0])
        rows_m.append([0, 0, 0] if i % 2 == 0 else [1, 1, 1])
    return select(batch_from_arrays(rows_t, rows_m), slice(None), torch.float32)


def conditional_poisson(rng, histories, events=6, swap_marks=False):
    times, marks = [], []
    for row in range(len(histories.time_seqs)):
        context = int(histories.type_seqs[row, -1])
        rate, probability = (8.0, 0.95) if context == 0 else (0.2, 0.05)
        if swap_marks:
            probability = 1 - probability
        times.append(np.cumsum(rng.exponential(1 / rate, events)))
        marks.append((rng.random(events) >= probability).astype(int))
    return batch_from_arrays(times, marks)


def conditional_hawkes_reference(rng, histories, mu, alpha, beta, events=6):
    """NumPy oracle initialized from every row's actual masked history.

    Decay all past excitation to the conditioning time, then output relative
    continuation times. Mark probabilities use the post-decay intensities.
    """
    mu, alpha, beta = map(np.asarray, (mu, alpha, beta))
    rows_t, rows_m = [], []
    for row in range(len(histories.time_seqs)):
        mask = histories.valid_event_mask[row]
        ht = histories.time_seqs[row][mask].double().numpy()
        hm = histories.type_seqs[row][mask].numpy()
        end = float(ht[-1])
        state = np.zeros_like(alpha)
        for time, mark in zip(ht, hm):
            state[:, mark] += (
                alpha[:, mark] * beta[:, mark] * np.exp(-beta[:, mark] * (end - time))
            )
        times, marks, t = [], [], 0.0
        while len(times) < events:
            bound = float((mu + state.sum(axis=1)).sum())
            dt = rng.exponential(1 / bound)
            t += dt
            state *= np.exp(-beta * dt)
            intensity = mu + state.sum(axis=1)
            if rng.random() * bound <= intensity.sum():
                mark = int(rng.choice(len(mu), p=intensity / intensity.sum()))
                times.append(t)
                marks.append(mark)
                state[:, mark] += alpha[:, mark] * beta[:, mark]
        rows_t.append(times)
        rows_m.append(marks)
    return batch_from_arrays(rows_t, rows_m)


def swap_orders(n, permutations, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.stack(
            [torch.arange(2 * n)]
            + [torch.cat(pair_swap_indices(n)) for _ in range(permutations)]
        )


def cached_paired_statistics(test, x, y, orders):
    """SIG-only cache; pairwise min horizons commute with permitted swaps.

    Public-API equivalence is checked in unit tests and the first trial of each
    experiment/batch. This shortcut is not asserted for arbitrary kernels.
    """
    n = len(x.time_seqs)
    x, y = test._truncate_to_min_time(x, y)
    pool = test._concat_batches(x, y)
    gram = test.kernel.compute_gram_matrix(pool, pool)
    return torch.stack([gram_statistic(gram, order[:n], order[n:]) for order in orders])


def exact_product_group_statistics(test, batches):
    """Exhaustive independent swaps across batches: oracle for aggregated ranks."""
    block_values = []
    for x, y in batches:
        n = len(x.time_seqs)
        rows = torch.arange(n)
        orders = torch.stack(
            [
                torch.cat(
                    (rows + n * torch.tensor(bits), rows + n * (1 - torch.tensor(bits)))
                )
                for bits in itertools.product((0, 1), repeat=n)
            ]
        )
        block_values.append(cached_paired_statistics(test, x, y, orders))
    return torch.tensor(
        [
            sum(float(block_values[i][j]) for i, j in enumerate(indices))
            for indices in itertools.product(*(range(len(v)) for v in block_values))
        ],
        dtype=torch.float64,
    )


def run(
    output, trials=500, power_trials=200, permutations=99, steps=200, seed=20261017
):
    if min(trials, power_trials, permutations, steps) < 1:
        raise ValueError("Counts must be positive")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    train = marked_reference(np.random.default_rng(seed), "hawkes", 256)
    models, fits = {}, []
    for i, name in enumerate(("Hawkes", "NHP")):
        models[name], fit = fit_model(name, train, output, seed + 1 + i, steps)
        fits.append(fit)
    report = dict(
        status="in_progress",
        seed=seed,
        code=code_provenance(__file__),
        fits=fits,
        trials_H0=trials,
        trials_H1=power_trials,
        permutations=permutations,
        batch_sizes=[8, 5, 3],
        events_per_continuation=6,
        marks=2,
        aggregation="sum_statistics_product_randomization_v1",
        results=[],
        scope="Independent pairs/batches, fixed heterogeneous histories, independently fitted frozen checkpoints. Shared individuals, overlapping batches and rolling histories are excluded.",
    )

    def save():
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    for case_id, case in enumerate(
        (
            "context_poisson",
            "Hawkes_oracle",
            "NHP_internal",
            "independent_batches_poisson",
            "context_poisson_mark_alternative",
            "overlapping_batches_control",
        )
    ):
        count = power_trials if "alternative" in case else trials
        rng = np.random.default_rng(np.random.SeedSequence([seed, 100, case_id]))
        conditional = case not in (
            "independent_batches_poisson",
            "overlapping_batches_control",
        )
        cls = ConditionalMMDTwoSampleTest if conditional else MMDTwoSampleTest
        test = cls(SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2), permutations)
        p_values = dict(single_first_batch=[], aggregated=[])
        if case == "context_poisson":
            p_values["invalid_global_shuffle_control"] = []
        import hashlib

        raw_digest, order_digest, global_order_digest = (
            hashlib.sha256(),
            hashlib.sha256(),
            hashlib.sha256(),
        )
        for trial in range(count):
            test.reset_accumulators()
            sizes = (
                [8, 8, 8]
                if case == "overlapping_batches_control"
                else report["batch_sizes"]
            )
            reused = None
            for block_id, n in enumerate(sizes):
                h = heterogeneous_histories(n)
                order_seed = seed + 1000000 * case_id + 1000 * trial + block_id
                if case.startswith("context_poisson"):
                    x = conditional_poisson(rng, h)
                    y = conditional_poisson(rng, h, swap_marks="alternative" in case)
                elif case == "overlapping_batches_control":
                    if reused is None:
                        reused = (
                            marked_reference(rng, "poisson", n),
                            marked_reference(rng, "poisson", n),
                        )
                    x, y = reused
                elif case == "independent_batches_poisson":
                    mu = np.array([1.3, 0.7]) * [0.1, 1, 6][block_id]
                    x = marked_reference(rng, "poisson", n, mu=mu)
                    y = marked_reference(rng, "poisson", n, mu=mu)
                else:
                    name = "Hawkes" if case == "Hawkes_oracle" else "NHP"
                    model = models[name]
                    torch.manual_seed(order_seed + 10000000)
                    with torch.no_grad():
                        y = model.simulate(h, num_events_to_simulate=6)
                        if name == "NHP":
                            x = model.simulate(h, num_events_to_simulate=6)
                    if name == "Hawkes":
                        params = next(f for f in fits if f["model"] == name)[
                            "effective_parameters"
                        ]
                        x = conditional_hawkes_reference(rng, h, **params)
                raw_digest.update(
                    (fingerprint(h) + fingerprint(x) + fingerprint(y)).encode()
                )
                if conditional:
                    orders = swap_orders(n, permutations, order_seed)
                    stats = cached_paired_statistics(test, x, y, orders)
                else:
                    generator = torch.Generator().manual_seed(order_seed)
                    orders = torch.stack(
                        [torch.arange(2 * n)]
                        + [
                            torch.randperm(2 * n, generator=generator)
                            for _ in range(permutations)
                        ]
                    )
                    stats = paired_statistics(test, x, y, orders)[
                        "recomputed_pairwise_truncation"
                    ]
                order_digest.update(orders.numpy().tobytes())
                if trial == 0:
                    # Replay all null draws through actual production API before
                    # accepting the cache for this model and batch size.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(order_seed)
                        result = test.compute_statistics(
                            x,
                            y,
                            accumulate=False,
                            **({"histories": h} if conditional else {}),
                        )
                    torch.testing.assert_close(stats[0], result["observed_statistic"])
                    torch.testing.assert_close(stats[1:], result["permuted_statistics"])
                    assert float(monte_carlo_p_value(stats[0], stats[1:])) == float(
                        result["p_value"]
                    )
                test._accumulate(stats[0], stats[1:])
                if block_id == 0:
                    p_values["single_first_batch"].append(
                        float(monte_carlo_p_value(stats[0], stats[1:]))
                    )
                    if case == "context_poisson":
                        generator = torch.Generator().manual_seed(order_seed + 50000000)
                        global_orders = torch.stack(
                            [torch.arange(2 * n)]
                            + [
                                torch.randperm(2 * n, generator=generator)
                                for _ in range(permutations)
                            ]
                        )
                        global_stats = paired_statistics(test, x, y, global_orders)[
                            "recomputed_pairwise_truncation"
                        ]
                        global_order_digest.update(global_orders.numpy().tobytes())
                        p_values["invalid_global_shuffle_control"].append(
                            float(
                                monte_carlo_p_value(global_stats[0], global_stats[1:])
                            )
                        )
            p_values["aggregated"].append(float(test.get_final_p_value()))
            if (trial + 1) % 50 == 0:
                print(f"{case}: {trial+1}/{count}", flush=True)
        for method, values in p_values.items():
            rejected = sum(p <= 0.05 for p in values)
            result = dict(
                experiment=case,
                method=method,
                hypothesis="H1" if "alternative" in case else "H0",
                trials=count,
                rejections=rejected,
                rejection_rate=rejected / count,
                wilson_95=wilson_interval(rejected, count),
                p_values=values,
                raw_data_sha256=raw_digest.hexdigest(),
                permutations_sha256=(
                    global_order_digest.hexdigest()
                    if method == "invalid_global_shuffle_control"
                    else order_digest.hexdigest()
                ),
                batch_sizes=sizes,
                validity=(
                    "invalid_dependence_control"
                    if case == "overlapping_batches_control" and method == "aggregated"
                    else (
                        "invalid_history_exchangeability_control"
                        if method == "invalid_global_shuffle_control"
                        else "declared_H0_assumptions"
                    )
                ),
                public_api_all_draws_checked_for_first_trial=True,
            )
            report["results"].append(result)
            print(
                f"{case}/{method}: {rejected}/{count}, CI {result['wilson_95']}",
                flush=True,
            )
        save()
    report["status"] = "completed"
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/conditional-calibration")
    )
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--power-trials", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261017)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
