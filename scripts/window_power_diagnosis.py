"""Opt-in paired power diagnosis; no model or production kernel is changed."""

import argparse
import hashlib
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
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import heterogeneous_histories, swap_orders
from scripts.extended_scientific_validation import code_provenance
from scripts.fixed_window_calibration import window_continuations
from scripts.integration_suite import write_json
from scripts.scientific_calibration import wilson_interval
from scripts.stopped_conditional_calibration import (
    encoded_last_state,
    load_frozen_model,
    numpy_hazard,
)

ARCHIVES = Path("docs/validation/likelihood-exactness-20261014")
VARIANTS = ("signature8", "signature64", "count", "first_time")


def select(batch, start, stop):
    return Batch(
        *(
            getattr(batch, key)[start:stop]
            for key in ("time_seqs", "time_delta_seqs", "type_seqs", "valid_event_mask")
        )
    )


def summary_features(batch, horizon, kind):
    """Fixed features; an empty window has censored first time T, not termination."""
    if kind == "count":
        return torch.stack(
            [
                ((batch.type_seqs == mark) & batch.valid_event_mask).sum(-1)
                for mark in range(2)
            ],
            -1,
        ).double()
    if kind != "first_time":
        raise ValueError("Unknown diagnostic feature")
    return (
        batch.time_seqs.double()
        .masked_fill(~batch.valid_event_mask, horizon)
        .min(-1)
        .values[:, None]
        / horizon
    )


def feature_gram(batch, horizon, kind):
    features = summary_features(batch, horizon, kind)
    # Fixed before evaluation: one event for count, T/10 for first-event time.
    bandwidth = 1.0 if kind == "count" else 0.1
    return torch.exp(-0.5 * torch.cdist(features, features).square() / bandwidth**2)


def run(
    output,
    trials=100,
    permutations=99,
    seed=2026101407,
    model_name=None,
    epoch_number=None,
):
    if min(trials, permutations) < 1:
        raise ValueError("Positive counts required")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    horizon = 2.0
    report = dict(
        status="in_progress",
        code=code_provenance(__file__),
        seed=seed,
        trials_per_hypothesis=trials,
        permutations=permutations,
        horizon=horizon,
        plan="Four archived checkpoints; H0 and twice_speed H1; paired identical datasets/orders across four kernels and 32/128 independent pairs. Fixed history kernel signature8. No model, production kernel, alpha or bandwidth tuning.",
        variants=list(VARIANTS),
        sample_sizes=[32, 128],
        h0_family_size=32,
        h0_excess_threshold=0.01 / 32,
        scope="Exploratory diagnosis with predeclared comparisons, shared frozen encoder and NumPy/SciPy decoder. Independent trials, nested 32/128 samples; not a production kernel selection or general model ranking.",
        models=[],
        selected_model=model_name,
        selected_epoch=epoch_number,
    )
    write_json(output / "plan.json", report)
    sigs = {
        points: SIGKernel(RBFKernel(), "counting_grid", points, 0, 2)
        for points in (8, 64)
    }
    history = heterogeneous_histories(128)
    for name in ("FullyNN", "RMTPP"):
        if model_name is not None and name != model_name:
            continue
        for epoch in (1, 2):
            if epoch_number is not None and epoch != epoch_number:
                continue
            model, fit = load_frozen_model(
                name,
                output / f"{name}-{epoch}",
                ARCHIVES / f"{name}_{epoch}_frozen_parameters.json",
            )
            frozen = {k: v.clone() for k, v in model.state_dict().items()}
            item = dict(
                model=name,
                epoch=epoch,
                fit=fit,
                initial_history=[],
                results=[],
                observations={},
            )
            for hidden in encoded_last_state(model, select(history, 0, 2)):
                hazard, _, limit = numpy_hazard(model, hidden)
                item["initial_history"].append(
                    dict(
                        hazard_limit=float(limit) if np.isfinite(limit) else None,
                        permanent_stop_probability=float(np.exp(-limit)),
                        event_probability_T=float(-np.expm1(-hazard(horizon))),
                        event_probability_2T=float(-np.expm1(-hazard(2 * horizon))),
                        event_probability_4T=float(-np.expm1(-hazard(4 * horizon))),
                    )
                )
            for case in ("H0", "twice_speed"):
                pvalues = {(v, n): [] for v in VARIANTS for n in (32, 128)}
                observations = {
                    label: {"counts": [], "first_times": [], "event_times": []}
                    for label in ("x", "y")
                }
                collisions = {points: [0, 0] for points in (8, 64)}
                digest = hashlib.sha256()
                for trial in range(trials):
                    sample_seed = (
                        seed
                        + (name == "RMTPP") * 100000000
                        + epoch * 10000000
                        + (case != "H0") * 1000000
                        + trial * 1000
                    )
                    torch.manual_seed(sample_seed)
                    y = window_continuations(model, history, horizon)
                    x = window_continuations(
                        model,
                        history,
                        horizon,
                        rng=np.random.default_rng(sample_seed + 500000000),
                        speed=2.0 if case != "H0" else 1.0,
                    )
                    for label, batch in (("x", x), ("y", y)):
                        for key in ("time_seqs", "type_seqs", "valid_event_mask"):
                            digest.update(getattr(batch, key).numpy().tobytes())
                        observations[label]["counts"].extend(
                            batch.valid_event_mask.sum(-1).tolist()
                        )
                        observations[label]["first_times"].extend(
                            (
                                summary_features(batch, horizon, "first_time")[:, 0]
                                * horizon
                            ).tolist()
                        )
                        observations[label]["event_times"].extend(
                            batch.time_seqs[batch.valid_event_mask].tolist()
                        )
                    summed = {
                        v: torch.zeros(permutations + 1, dtype=torch.double)
                        for v in VARIANTS
                    }
                    offset = 0
                    for group in range(4):
                        for block, size in enumerate((16, 11, 5)):
                            h, bx, by = (
                                select(b, offset, offset + size)
                                for b in (history, x, y)
                            )
                            test = ConditionalMMDTwoSampleTest(
                                sigs[8], permutations, observation_horizon=horizon
                            )
                            pool = test._concat_batches(bx, by)
                            hg = sigs[8].compute_gram_matrix(h, h)
                            distance = (
                                hg.diag()[:, None] + hg.diag()[None, :] - 2 * hg
                            ).clamp_min(0)
                            weight = torch.exp(-0.5 * distance).repeat(2, 2)
                            orders = swap_orders(
                                size, permutations, sample_seed + 200000000 + offset
                            )
                            for variant in VARIANTS:
                                if variant.startswith("signature"):
                                    points = int(variant.removeprefix("signature"))
                                    kernel = sigs[points]
                                    gram = kernel.compute_gram_matrix(pool, pool)
                                    # Pure, coupled time compression keeps marks/counts fixed.
                                    compressed = Batch(
                                        by.time_seqs / 2,
                                        by.time_delta_seqs / 2,
                                        by.type_seqs,
                                        by.valid_event_mask,
                                    )
                                    a, b = kernel._prepare_kernel(by, compressed)
                                    collisions[points][0] += int(
                                        (a == b).all(-1).all(-1).sum()
                                    )
                                    collisions[points][1] += size
                                else:
                                    gram = feature_gram(pool, horizon, variant)
                                gram = gram * weight
                                stats = torch.stack(
                                    [
                                        paired_joint_statistic(
                                            gram, order[:size], order[size:]
                                        )
                                        for order in orders
                                    ]
                                )
                                if (
                                    trial == group == block == 0
                                    and variant == "signature8"
                                ):
                                    torch.testing.assert_close(
                                        gram,
                                        test.joint_kernel_matrix(h, bx, by),
                                        rtol=0,
                                        atol=0,
                                    )
                                summed[variant] += stats
                            offset += size
                        if group in (0, 3):
                            n = 32 if group == 0 else 128
                            for v in VARIANTS:
                                pvalues[v, n].append(
                                    float(
                                        monte_carlo_p_value(summed[v][0], summed[v][1:])
                                    )
                                )
                    if (trial + 1) % 10 == 0:
                        print(name, epoch, case, trial + 1, trials, flush=True)
                for (variant, n), values in pvalues.items():
                    rejects = sum(p <= 0.05 for p in values)
                    excess = (
                        binomtest(rejects, trials, 0.05, alternative="greater").pvalue
                        if case == "H0"
                        else None
                    )
                    item["results"].append(
                        dict(
                            hypothesis=case,
                            variant=variant,
                            pairs=n,
                            rejections=rejects,
                            trials=trials,
                            rejection_rate=rejects / trials,
                            wilson_95=wilson_interval(rejects, trials),
                            h0_excess_binomial_p=excess,
                            h0_excess_detected=bool(excess < 0.01 / 32)
                            if excess is not None
                            else None,
                            p_values=values,
                        )
                    )
                item["observations"][case] = dict(
                    data_sha256=digest.hexdigest(),
                    pure_half_time_embedding_collision={
                        str(k): v[0] / v[1] for k, v in collisions.items()
                    },
                    samples={},
                )
                for label, values in observations.items():
                    counts = np.asarray(values["counts"])
                    item["observations"][case]["samples"][label] = dict(
                        sequences=len(counts),
                        mean_events=float(counts.mean()),
                        empty_fraction=float((counts == 0).mean()),
                        event_count_histogram={
                            str(int(k)): int(v)
                            for k, v in zip(*np.unique(counts, return_counts=True))
                        },
                        censored_first_time_mean=float(np.mean(values["first_times"])),
                        event_time_quantiles=np.quantile(
                            values["event_times"], [0.1, 0.5, 0.9]
                        ).tolist()
                        if values["event_times"]
                        else [],
                    )
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
            item["model_frozen_verified"] = True
            report["models"].append(item)
            write_json(output / "report.json", report)
    report["status"] = "completed"
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--seed", type=int, default=2026101407)
    parser.add_argument("--model-name", choices=("FullyNN", "RMTPP"))
    parser.add_argument("--epoch-number", type=int, choices=(1, 2))
    run(**vars(parser.parse_args()))
