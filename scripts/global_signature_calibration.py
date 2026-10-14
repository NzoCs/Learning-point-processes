"""Opt-in global conditional signature MMD on complete fixed-window futures."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test import (
    paired_joint_statistic,
)
from scripts.conditional_calibration import heterogeneous_histories, swap_orders
from scripts.extended_scientific_validation import code_provenance
from scripts.fixed_window_calibration import window_continuations
from scripts.integration_suite import write_json
from scripts.scientific_calibration import wilson_interval
from scripts.stopped_conditional_calibration import load_frozen_model
from scripts.window_first_event_power import p_value
from scripts.window_power_diagnosis import ARCHIVES, select

VARIANTS = ("global", "blocks_global_normalization", "blocks_local_normalization")
SIZES = (32, 128, 512, 2048)


def compressed_encoding(kernel, batch):
    """Exact duplicate-path compression AFTER the usual pooled normalization.

    No quantization, approximation or event filtering beyond the production
    signature embedding. Retain every row's inverse index and multiplicity.
    """
    paths, other = kernel._prepare_kernel(batch, batch)
    if not torch.equal(paths, other):
        raise AssertionError("Self-kernel paths disagree")
    unique, inverse = torch.unique(paths, dim=0, return_inverse=True)
    gram = kernel.kernel.compute_Gram(unique, unique).detach().cpu().numpy()
    return gram, inverse.cpu().numpy()


def signature_delta(kernel, histories, x, y, horizon):
    n = len(x.time_seqs)
    if n < 2 or len(y.time_seqs) != n or len(histories.time_seqs) != n:
        raise ValueError("At least two matched history/future pairs required")
    test = ConditionalMMDTwoSampleTest(kernel, 1, observation_horizon=horizon)
    test._validate_window_continuation(x)
    test._validate_window_continuation(y)
    pool = test._concat_batches(x, y)
    future, inverse = compressed_encoding(kernel, pool)
    a, b = inverse[:n], inverse[n:]
    delta = future[np.ix_(a, a)].copy()
    delta += future[np.ix_(b, b)]
    delta -= future[np.ix_(a, b)]
    delta -= future[np.ix_(b, a)]
    history, hi = compressed_encoding(kernel, histories)
    distance = np.maximum(
        history.diagonal()[:, None] + history.diagonal()[None, :] - 2 * history, 0.0
    )
    weights = np.exp(-0.5 * distance)
    delta *= weights[np.ix_(hi, hi)]
    return delta, dict(
        unique_future_paths=len(future),
        total_future_paths=2 * n,
        unique_histories=len(history),
    )


def delta_statistics(delta, signs):
    n = len(delta)
    if delta.shape != (n, n) or signs.shape[1] != n or n < 2:
        raise ValueError("Square delta and matched swap signs required")
    return (
        np.einsum("pi,ij,pj->p", signs, delta, signs, optimize=True) - np.trace(delta)
    ) / (n * (n - 1))


def blocks(n):
    if n < 32 or n % 32:
        raise ValueError("The diagnostic uses complete 16/11/5 groups")
    offset = 0
    while offset < n:
        for size in (16, 11, 5):
            yield offset, offset + size
            offset += size


def verify_public_statistics(kernel, h, x, y, delta, signs, horizon):
    """Check all swaps against the public Gram, not a separate new estimator."""
    test = ConditionalMMDTwoSampleTest(
        kernel, len(signs) - 1, observation_horizon=horizon
    )
    gram = test.joint_kernel_matrix(h, x, y)
    n = len(x.time_seqs)
    expected_delta = gram[:n, :n] + gram[n:, n:] - gram[:n, n:] - gram[n:, :n]
    np.testing.assert_allclose(delta, expected_delta.numpy(), rtol=1e-12, atol=1e-13)
    values = []
    for sign in signs:
        swap = torch.tensor((sign < 0).astype(int))
        rows = torch.arange(n)
        values.append(
            float(paired_joint_statistic(gram, rows + n * swap, rows + n * (1 - swap)))
        )
    np.testing.assert_allclose(
        delta_statistics(delta, signs), values, rtol=1e-10, atol=1e-12
    )


def run(
    model_name,
    epoch_number,
    output,
    trials=300,
    power_trials=200,
    permutations=99,
    max_pairs=2048,
    hypothesis=None,
    seed=2026101427,
):
    if (
        min(trials, power_trials, permutations) < 1
        or max_pairs not in SIZES
        or hypothesis not in (None, "H0", "twice_speed")
    ):
        raise ValueError("Positive trial counts and supported sample size required")
    if (output / "report.json").exists():
        raise FileExistsError("Use a fresh output folder to preserve existing evidence")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    horizon = 2.0
    sizes = [n for n in SIZES if n <= max_pairs]
    report = dict(
        status="in_progress",
        code=code_provenance(__file__),
        model=model_name,
        epoch=epoch_number,
        seed=seed,
        trials_H0=trials,
        trials_H1=power_trials,
        permutations=permutations,
        horizon=horizon,
        observation_clock_dtype="float64",
        model_input_dtype="float32; gaps computed before casting the observation clock",
        sample_sizes=sizes,
        selected_hypothesis=hypothesis,
        variants=list(VARIANTS),
        kernel=dict(
            type="signature",
            static="RBF",
            sigma=1.0,
            grid_points=8,
            dyadic_order=0,
            marks=2,
            backend="pysiglib",
        ),
        h0_family_size=48,
        h0_excess_threshold=0.01 / 48,
        plan="Four archived frozen checkpoints; 300 H0 trials and 200 twice-speed H1 trials, 99 within-history paired swaps; 32/128/512/2048 independent pairs. Three prespecified statistics on identical full-window data and identical signs: global paired U-MMD, 16/11/5 block sum with global normalization, historical block sum with local normalization. Original signature8 kernel and product history weights. Exact duplicate-path compression only; no fitted-model or production change.",
        scope="Complete futures inside a fully observed fixed T=2 window, including empty windows; no inference of permanent termination. Production simulator versus independent NumPy/SciPy decoder/inversion/RNG, shared frozen Torch encoder. Two repeated heterogeneous histories, two checkpoints per model; no fitting uncertainty or CUDA/Ruche validation.",
        results=[],
        observations={},
    )
    write_json(output / "plan.json", report)
    model, fit = load_frozen_model(
        model_name,
        output / "model",
        ARCHIVES / f"{model_name}_{epoch_number}_frozen_parameters.json",
    )
    report["fit"] = fit
    frozen = {key: value.clone() for key, value in model.state_dict().items()}
    h = heterogeneous_histories(max_pairs)
    kernel = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)
    cases = (hypothesis,) if hypothesis else ("H0", "twice_speed")
    for case in cases:
        count = trials if case == "H0" else power_trials
        values = {(variant, n): [] for variant in VARIANTS for n in sizes}
        digest = hashlib.sha256()
        counts = {label: [] for label in ("x", "y")}
        unique_counts = {n: [] for n in sizes}
        started = time.perf_counter()
        for trial in range(count):
            sample_seed = (
                seed
                + (model_name == "RMTPP") * 100000000
                + epoch_number * 10000000
                + (case != "H0") * 1000000
                + trial * 1000
            )
            torch.manual_seed(sample_seed)
            y = window_continuations(model, h, horizon, clock_dtype=torch.float64)
            x = window_continuations(
                model,
                h,
                horizon,
                rng=np.random.default_rng(sample_seed + 500000000),
                speed=2.0 if case != "H0" else 1.0,
                clock_dtype=torch.float64,
            )
            for label, batch in (("x", x), ("y", y)):
                for key in ("time_seqs", "type_seqs", "valid_event_mask"):
                    digest.update(getattr(batch, key).numpy().tobytes())
                counts[label].extend(batch.valid_event_mask.sum(-1).tolist())
            orders = swap_orders(max_pairs, permutations, sample_seed + 200000000)
            signs = 1.0 - 2.0 * (orders[:, :max_pairs].numpy() // max_pairs)
            local_sum = np.zeros(permutations + 1)
            local_statistics = {}
            for start, stop in blocks(max_pairs):
                dh, dx, dy = (select(batch, start, stop) for batch in (h, x, y))
                delta, _ = signature_delta(kernel, dh, dx, dy, horizon)
                local_sum += delta_statistics(delta, signs[:, start:stop])
                if stop in sizes:
                    local_statistics[stop] = local_sum.copy()
            for n in sizes:
                dh, dx, dy = (select(batch, 0, n) for batch in (h, x, y))
                delta, meta = signature_delta(kernel, dh, dx, dy, horizon)
                unique_counts[n].append(meta["unique_future_paths"])
                global_stats = delta_statistics(delta, signs[:, :n])
                normalized_blocks = sum(
                    (
                        delta_statistics(
                            delta[start:stop, start:stop], signs[:, start:stop]
                        )
                        for start, stop in blocks(n)
                    ),
                    np.zeros(permutations + 1),
                )
                if trial == 0 and n == 32:
                    verify_public_statistics(
                        kernel, dh, dx, dy, delta, signs[:, :n], horizon
                    )
                for variant, stats in (
                    ("global", global_stats),
                    ("blocks_global_normalization", normalized_blocks),
                    ("blocks_local_normalization", local_statistics[n]),
                ):
                    values[variant, n].append(p_value(stats))
            if (trial + 1) % 10 == 0:
                print(
                    model_name,
                    epoch_number,
                    "global-signature",
                    case,
                    trial + 1,
                    count,
                    round(time.perf_counter() - started, 1),
                    "seconds",
                    flush=True,
                )
        for (variant, n), pvalues in values.items():
            rejected = sum(p <= 0.05 for p in pvalues)
            excess = (
                binomtest(rejected, count, 0.05, alternative="greater").pvalue
                if case == "H0"
                else None
            )
            report["results"].append(
                dict(
                    hypothesis=case,
                    variant=variant,
                    pairs=n,
                    trials=count,
                    rejections=rejected,
                    rejection_rate=rejected / count,
                    wilson_95=wilson_interval(rejected, count),
                    h0_excess_binomial_p=excess,
                    h0_excess_detected=(
                        bool(excess < 0.01 / 48) if excess is not None else None
                    ),
                    p_values=pvalues,
                )
            )
        report["observations"][case] = dict(
            data_sha256=digest.hexdigest(),
            elapsed_seconds=time.perf_counter() - started,
            exact_duplicate_compression={
                str(n): dict(
                    min=int(min(v)),
                    mean=float(np.mean(v)),
                    max=int(max(v)),
                    pooled_rows=2 * n,
                )
                for n, v in unique_counts.items()
            },
            samples={
                label: dict(
                    sequences=len(v),
                    mean_events=float(np.mean(v)),
                    empty_fraction=float(np.mean(np.array(v) == 0)),
                )
                for label, v in counts.items()
            },
            public_gram_and_all_permutation_statistics_verified_on_first_trial=True,
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
        write_json(output / "report.json", report)
    report.update(status="completed", model_frozen_verified=True)
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=("FullyNN", "RMTPP"), required=True)
    parser.add_argument("--epoch-number", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--power-trials", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--max-pairs", type=int, choices=SIZES, default=2048)
    parser.add_argument("--hypothesis", choices=("H0", "twice_speed"))
    parser.add_argument("--seed", type=int, default=2026101427)
    run(**vars(parser.parse_args()))
