"""Paired, opt-in kernel/clock/quota ablation; never modifies fitted models."""

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
from new_ltpp.evaluation.statistical_testing.statistical_tests.global_conditional_mmd_test import (
    copy_rows,
)
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import swap_orders
from scripts.extended_scientific_validation import code_provenance
from scripts.global_signature_calibration import compressed_encoding, delta_statistics
from scripts.integration_suite import write_json
from scripts.pipeline_protocol_calibration import (
    analytic_continuations,
    histories_for_trial,
    model_and_config,
    state_hash,
)
from scripts.scientific_calibration import wilson_interval
from scripts.window_first_event_power import p_value

VARIANTS = (
    "grid8_pool",
    "grid32_pool",
    "grid8_physical",
    "grid32_physical",
    "events_pool",
    "events_physical",
    "first_time_physical",
    "counts",
)
SEED = 2026101561


def concat(x, y):
    # The production helper has no instance-dependent state.
    return ConditionalMMDTwoSampleTest._concat_batches(None, x, y)


def quota_view(batch, quota):
    """An eventual stop beyond the quota must not be revealed at a lower quota."""
    n = len(batch.time_seqs)
    width = min(quota, batch.time_seqs.shape[1])
    t = torch.zeros(n, quota, dtype=batch.time_seqs.dtype)
    d = torch.zeros_like(t)
    m = torch.full((n, quota), 2, dtype=torch.long)
    valid = torch.zeros(n, quota, dtype=torch.bool)
    for target, source in (
        (t, batch.time_seqs),
        (d, batch.time_delta_seqs),
        (m, batch.type_seqs),
        (valid, batch.valid_event_mask),
    ):
        target[:, :width] = source[:, :width]
    counts = valid.sum(1)
    flag = counts < quota
    if (flag & ~batch.no_next_event_mask).any():
        raise ValueError(
            "Insufficient trajectory: quota not reached and no actual stop"
        )
    return Batch(t, d, m, valid, no_next_event_mask=flag)


def speed_view(batch):
    out = copy_rows(batch)
    out.time_seqs = out.time_seqs / 2
    out.time_delta_seqs = out.time_delta_seqs / 2
    return out


def event_paths(pool, denominator):
    """Retain every jump, with shared endpoint and duplicate-endpoint padding."""
    n, width = pool.time_seqs.shape
    endpoint = float(pool.time_seqs.masked_fill(~pool.valid_event_mask, 0).max())
    paths = torch.zeros(n, 2 * width + 2, 4, dtype=torch.float64)
    for row in range(n):
        mask = pool.valid_event_mask[row]
        times, marks = pool.time_seqs[row, mask], pool.type_seqs[row, mask]
        count = len(times)
        value = torch.zeros(4, dtype=torch.float64)
        position = 1
        for t, mark in zip(times, marks):
            value[0] = float(t) / denominator
            paths[row, position] = value
            value[1] += 1 / (max(count - 1, 1) + 1e-8)
            value[2 + int(mark)] += 1
            paths[row, position + 1] = value
            position += 2
        value[0] = endpoint / denominator
        paths[row, position:] = value
    return paths


def gram_from_paths(kernel, paths):
    unique, inverse = torch.unique(paths, dim=0, return_inverse=True)
    gram = kernel.kernel.compute_Gram(unique, unique).detach().numpy()
    ix = inverse.numpy()
    return gram[np.ix_(ix, ix)], len(unique)


def history_weights(kernel, h):
    gram, inv = compressed_encoding(kernel, h)
    distance = np.maximum(
        gram.diagonal()[:, None] + gram.diagonal()[None, :] - 2 * gram, 0
    )
    return np.exp(-0.5 * distance)[np.ix_(inv, inv)]


def future_gram(kernel, pool, variant, physical_scale):
    maximum = float(pool.time_seqs.masked_fill(~pool.valid_event_mask, 0).max()) + 1e-8
    if variant.startswith("grid"):
        paths, _ = kernel._prepare_kernel(pool, pool)
        if variant.endswith("physical"):
            # Preserve every bin/event; change ONLY the time-channel scale.
            # Dividing input times by a fixed constant with a [0,1] grid would
            # silently discard late events and confound normalization/censoring.
            paths[:, :, 0] *= maximum / physical_scale
        return gram_from_paths(kernel, paths)
    if variant.startswith("events"):
        scale = maximum if variant.endswith("pool") else physical_scale
        return gram_from_paths(kernel, event_paths(pool, scale))
    counts = pool.valid_event_mask.sum(1).numpy()
    if variant == "counts":
        features = np.stack(
            [
                ((pool.type_seqs == mark) & pool.valid_event_mask).sum(1).numpy()
                for mark in (0, 1)
            ],
            axis=1,
        )
    elif variant in ("first_time_physical", "first_time_centered"):
        first = (
            pool.time_seqs.masked_fill(~pool.valid_event_mask, float("inf"))
            .min(1)
            .values.numpy()
        )
        # Separate existence channel: an absent event is not assigned a time.
        transformed = np.zeros_like(first, dtype=np.float64)
        np.log1p(first / physical_scale, out=transformed, where=counts > 0)
        features = np.stack([transformed, counts > 0], axis=1)
    else:
        raise ValueError(variant)
    distance = ((features[:, None] - features[None, :]) ** 2).sum(-1)
    gram = np.exp(-0.5 * distance)
    if variant == "first_time_centered":
        present = counts > 0
        if not present.any():
            return np.zeros_like(gram), 1
        # Center RKHS features on the pooled nonempty observations. Empty
        # futures have the zero feature, not an invented event time. This is
        # a PSD Gram (a feature translation followed by a presence gate), and
        # preprocessing depends only on the pool, hence commutes with swaps.
        mean = gram[:, present].mean(1)
        gram = (
            gram - mean[:, None] - mean[None, :] + gram[np.ix_(present, present)].mean()
        )
        gram *= present[:, None] * present[None, :]
    return gram, len(np.unique(features, axis=0))


def statistics(kernel, hweight, x, y, signs, variant, physical_scale):
    pool = concat(x, y)
    future, unique = future_gram(kernel, pool, variant, physical_scale)
    terminal = torch.cat((x.no_next_event_mask, y.no_next_event_mask)).double().numpy()
    future *= np.exp(-0.5 * (terminal[:, None] - terminal[None, :]) ** 2)
    n = len(x.time_seqs)
    delta = (
        future[:n, :n] + future[n:, n:] - future[:n, n:] - future[n:, :n]
    ) * hweight
    return delta_statistics(delta, signs), unique, delta


def mechanism(kernel, x, y):
    pool = concat(x, y)
    paths, _ = kernel._prepare_kernel(pool, pool)
    n = len(x.time_seqs)
    accelerated = concat(speed_view(x), y)
    maximum = (
        pool.time_seqs.masked_fill(~pool.valid_event_mask, 0).max().double() + 1e-8
    )
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.utils import (
        _get_embedding,
    )

    accelerated_paths = _get_embedding(
        8,
        "counting_grid",
        2,
        accelerated.time_seqs.double() / maximum,
        accelerated.type_seqs,
        accelerated.valid_event_mask,
    )
    nonempty = x.valid_event_mask.any(1)
    unchanged = (paths[:n] == accelerated_paths[:n]).all(dim=(1, 2)) & nonempty
    valid_times = pool.time_seqs[pool.valid_event_mask]
    return dict(
        rows=n,
        empty_x=int((~nonempty).sum()),
        events=int(pool.valid_event_mask.sum()),
        events_first_bin=int((valid_times <= maximum / 7).sum()),
        nonempty_x=int(nonempty.sum()),
        acceleration_erased_rows=int(unchanged.sum()),
        pooled_max=float(maximum),
        median_event_time=float(valid_times.median()) if len(valid_times) else None,
    )


def run(name, output, h0=300, h1=200, pairs=128, centered_control=False):
    if min(h0, h1, pairs) < 2:
        raise ValueError("At least two trials/pairs required")
    torch.set_num_threads(1)
    variants = (
        ("first_time_physical", "first_time_centered") if centered_control else VARIANTS
    )
    base_seed = 2026101583 if centered_control else SEED
    family_size = 4 * len(variants)
    output = Path(output)
    if (output / "plan.json").exists():
        raise FileExistsError("Use a fresh output folder")
    plan = dict(
        model=name,
        status="in_progress",
        code=code_provenance(__file__),
        seed=base_seed,
        centered_followup=centered_control,
        H0_trials=h0,
        H1_trials=h1,
        pairs=pairs,
        quotas=[3, 6],
        variants=list(variants),
        permutations=99,
        alpha=0.05,
        family_size=family_size,
        family_excess_threshold=0.01 / family_size,
        scope="Same frozen current-pipeline checkpoint/config; native Y and independent NumPy/SciPy X, shared Torch encoder. Same trajectories and signs across quotas and variants; H1 halves X's complete path times without re-encoding. No model, sampler or production-kernel changes. Quotas only: no claim of a fully observed time window. CPU.",
        normalization="Physical grid variants rescale time channel only, keep pooled grid bins. Event paths keep all jumps and shared endpoint. Fixed scale is train data_info dtime_max, never selected using H1. History and terminal kernels always production defaults. First-time/count RBF controls use fixed bandwidth 1. All variants predeclared, report all; no winner tuning.",
    )
    write_json(output / "plan.json", plan)
    model, _, fit = model_and_config(name, output)
    plan["fit"] = fit
    original = state_hash(model)
    scale = float(model.dtime_max)
    plan["physical_scale"] = scale
    kernels = {g: SIGKernel(RBFKernel(), "counting_grid", g, 0, 2) for g in (8, 32)}
    observations = {q: [] for q in (3, 6)}
    values = {
        (case, q, variant): []
        for case in ("H0", "twice_speed")
        for q in (3, 6)
        for variant in variants
    }
    digest = hashlib.sha256()
    started = time.monotonic()
    errors = []
    # H0 and H1 paired on the same independently sampled futures. More H0
    # trials than H1; no resampling or selecting successful trials.
    for trial in range(max(h0, h1)):
        try:
            seed = base_seed + (name == "RMTPP") * 10000000 + trial * 1000
            h = histories_for_trial(pairs, np.random.default_rng(seed), trial)
            x = analytic_continuations(
                model, h, np.random.default_rng(seed + 100000000), 6
            )
            torch.manual_seed(seed + 200000000)
            ys = []
            for start in range(0, pairs, 64):
                ys.append(
                    model.simulate(
                        copy_rows(h, list(range(start, min(start + 64, pairs)))),
                        num_events_to_simulate=6,
                    )
                )
            y = ys[0]
            for part in ys[1:]:
                y = concat(y, part)
            for batch in (h, x, y):
                for key in ("time_seqs", "type_seqs", "valid_event_mask"):
                    digest.update(getattr(batch, key).numpy().tobytes())
            orders = swap_orders(pairs, 99, seed + 300000000)
            signs = 1.0 - 2.0 * (orders[:, :pairs].numpy() // pairs)
            weights = history_weights(kernels[8], h)
            for q in (3, 6):
                qx, qy = quota_view(x, q), quota_view(y, q)
                observations[q].append(mechanism(kernels[8], qx, qy))
                for case, active in (("H0", trial < h0), ("twice_speed", trial < h1)):
                    if not active:
                        continue
                    current = qx if case == "H0" else speed_view(qx)
                    for variant in variants:
                        kernel = kernels[32 if variant.startswith("grid32") else 8]
                        stats, unique, delta = statistics(
                            kernel, weights, current, qy, signs, variant, scale
                        )
                        if trial == 0 and case == "H0" and variant == "grid8_pool":
                            public = (
                                ConditionalMMDTwoSampleTest(
                                    kernel, 99, stopped_event_budget=q
                                )
                                .joint_kernel_matrix(h, current, qy)
                                .numpy()
                            )
                            expected = (
                                public[:pairs, :pairs]
                                + public[pairs:, pairs:]
                                - public[:pairs, pairs:]
                                - public[pairs:, :pairs]
                            )
                            np.testing.assert_allclose(
                                delta, expected, rtol=1e-10, atol=1e-12
                            )
                            np.testing.assert_allclose(
                                stats,
                                delta_statistics(expected, signs),
                                rtol=1e-10,
                                atol=1e-12,
                            )
                        values[case, q, variant].append(
                            dict(
                                p_value=p_value(stats),
                                observed=float(stats[0]),
                                unique_paths=unique,
                            )
                        )
        except Exception as exc:
            errors.append(dict(trial=trial, error=repr(exc)))
            # Preserve failed evidence; never silently condition on successes.
            break
        if (trial + 1) % 25 == 0:
            print(
                f"{name}: {trial + 1}/{max(h0, h1)} trials, {time.monotonic()-started:.1f}s",
                flush=True,
            )
    results = []
    for (case, q, variant), samples in values.items():
        ps = [s["p_value"] for s in samples]
        rejected = sum(p <= 0.05 for p in ps)
        result = dict(
            case=case,
            quota=q,
            variant=variant,
            n=len(ps),
            expected=h0 if case == "H0" else h1,
            rejected=rejected,
            rate=rejected / len(ps) if ps else None,
            ci95=wilson_interval(rejected, len(ps)) if ps else None,
            samples=samples,
        )
        if case == "H0" and ps:
            result["excess_p"] = float(
                binomtest(rejected, len(ps), 0.05, alternative="greater").pvalue
            )
        results.append(result)
    frozen = state_hash(model) == original
    plan.update(
        status="completed" if not errors and frozen else "failed",
        results=results,
        observations=observations,
        errors=errors,
        model_unchanged=frozen,
        trajectory_sha256=digest.hexdigest(),
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(output / "report.json", plan)
    if errors or not frozen:
        raise RuntimeError(f"Incomplete diagnostic: {errors}")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", choices=("FullyNN", "RMTPP"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h0", type=int, default=300)
    parser.add_argument("--h1", type=int, default=200)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--centered-control", action="store_true")
    args = parser.parse_args()
    run(args.model, args.output, args.h0, args.h1, args.pairs, args.centered_control)
