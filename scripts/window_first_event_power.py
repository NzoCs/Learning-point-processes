"""Stage-two sparse-window control on the exact frozen first-event law only."""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import binomtest

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)
from scripts.conditional_calibration import heterogeneous_histories
from scripts.extended_scientific_validation import code_provenance
from scripts.integration_suite import write_json
from scripts.scientific_calibration import wilson_interval
from scripts.stopped_conditional_calibration import (
    encoded_last_state,
    load_frozen_model,
    numpy_hazard,
)
from scripts.window_power_diagnosis import ARCHIVES


def p_value(statistics):
    values = torch.from_numpy(np.asarray(statistics, dtype=np.float64))
    return float(monte_carlo_p_value(values[0], values[1:]))


def pooled_first_time_statistics(x, y, history_weights, signs, horizon):
    """Exact same RBF product kernel, including pairs across former batch edges."""
    x, y = x / horizon, y / horizon
    delta = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 0.1**2)
    delta += np.exp(-0.5 * (y[:, None] - y[None, :]) ** 2 / 0.1**2)
    delta -= np.exp(-0.5 * (x[:, None] - y[None, :]) ** 2 / 0.1**2)
    delta -= np.exp(-0.5 * (y[:, None] - x[None, :]) ** 2 / 0.1**2)
    delta *= history_weights
    n = len(x)
    return (
        np.einsum("pi,ij,pj->p", signs, delta, signs, optimize=True) - np.trace(delta)
    ) / (n * (n - 1))


def signed_statistics(gram, signs):
    """Same paired U-statistic for all within-pair swaps, evaluated together."""
    n = signs.shape[1]
    delta = gram[:n, :n] + gram[n:, n:] - gram[:n, n:] - gram[n:, :n]
    return (np.einsum("pi,ij,pj->p", signs, delta, signs) - np.trace(delta)) / (
        n * (n - 1)
    )


def first_times(hazards, contexts, rng, horizon, speed):
    targets = rng.exponential(size=len(contexts))
    result = np.full(len(contexts), horizon)
    for context, hazard in enumerate(hazards):
        valid = np.flatnonzero(
            (contexts == context) & (targets < hazard(horizon * speed))
        )
        for row in valid:
            result[row] = (
                brentq(
                    lambda t: hazard(t) - targets[row], 0.0, horizon * speed, xtol=1e-12
                )
                / speed
            )
    return result


def run(model_name, epoch_number, output, trials=100, permutations=99, seed=2026101419):
    if min(trials, permutations) < 1:
        raise ValueError("Positive counts required")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    horizon = 2.0
    report = dict(
        status="in_progress",
        code=code_provenance(__file__),
        model=model_name,
        epoch=epoch_number,
        trials=trials,
        permutations=permutations,
        seed=seed,
        pairs=[32, 128, 512, 2048],
        plan="Stage three declared after stage-two targeted control. Exact NumPy/SciPy first-event law, same frozen parameters, two fixed histories. Same first-time RBF (bandwidth T/10) and signature8 history weights: 16/11/5 independent block aggregation versus pooled U-statistic, identical data and concatenated within-pair swaps. Additional targeted two-sided paired mean diagnostic. Production conservative numerical-tie rule for every p-value. No tuning or model change.",
        scope="This isolates the first event and sample size, not the full continuation or production simulator calibration; H0 uses two independent analytic-oracle samples. No events are discarded or conditioned to exist.",
        h0_family_size=24,
        h0_excess_threshold=0.01 / 24,
        results=[],
        analytical=[],
    )
    write_json(output / "plan.json", report)
    model, fit = load_frozen_model(
        model_name,
        output / "model",
        ARCHIVES / f"{model_name}_{epoch_number}_frozen_parameters.json",
    )
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    report["fit"] = fit
    h = heterogeneous_histories(2)
    hazards = [
        numpy_hazard(model, hidden)[0] for hidden in encoded_last_state(model, h)
    ]
    history = (
        SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)
        .compute_gram_matrix(h, h)
        .numpy()
    )
    weights = np.exp(
        -0.5
        * np.maximum(
            history.diagonal()[:, None] + history.diagonal()[None, :] - 2 * history, 0
        )
    )
    contexts = np.arange(2048) % 2
    for hazard in hazards:
        means = [
            quad(lambda t: np.exp(-hazard(speed * t)), 0.0, horizon, epsabs=1e-10)[0]
            for speed in (1.0, 2.0)
        ]
        report["analytical"].append(
            dict(
                event_probability_original=float(-np.expm1(-hazard(horizon))),
                event_probability_accelerated=float(-np.expm1(-hazard(2 * horizon))),
                censored_first_time_mean_original=means[0],
                censored_first_time_mean_accelerated=means[1],
            )
        )
    for case in ("H0", "twice_speed"):
        pvalues = {
            (variant, size): []
            for variant in ("first_time_rbf", "first_time_rbf_pooled", "paired_mean")
            for size in report["pairs"]
        }
        for trial in range(trials):
            sample_seed = (
                seed + epoch_number * 10000000 + (case != "H0") * 1000000 + trial * 1000
            )
            x = first_times(
                hazards,
                contexts,
                np.random.default_rng(sample_seed),
                horizon,
                2.0 if case != "H0" else 1.0,
            )
            y = first_times(
                hazards,
                contexts,
                np.random.default_rng(sample_seed + 500000000),
                horizon,
                1.0,
            )
            summed = np.zeros(permutations + 1)
            summed_mean = np.zeros(permutations + 1)
            offset = 0
            all_signs = []
            for group in range(64):
                for size in (16, 11, 5):
                    cx = contexts[offset : offset + size]
                    bx, by = x[offset : offset + size], y[offset : offset + size]
                    pool = np.concatenate((bx, by)) / horizon
                    gram = np.exp(-0.5 * (pool[:, None] - pool[None, :]) ** 2 / 0.1**2)
                    gram *= np.tile(weights[cx[:, None], cx[None, :]], (2, 2))
                    signs = np.concatenate(
                        (
                            np.ones((1, size)),
                            1.0
                            - 2
                            * np.random.default_rng(
                                sample_seed + 200000000 + offset
                            ).integers(2, size=(permutations, size)),
                        ),
                        axis=0,
                    )
                    summed += signed_statistics(gram, signs)
                    summed_mean += signs @ (bx - by)
                    all_signs.append(signs)
                    offset += size
                if offset in report["pairs"]:
                    pvalues["first_time_rbf", offset].append(p_value(summed))
                    cx = contexts[:offset]
                    pooled = pooled_first_time_statistics(
                        x[:offset],
                        y[:offset],
                        weights[cx[:, None], cx[None, :]],
                        np.concatenate(all_signs, axis=1),
                        horizon,
                    )
                    pvalues["first_time_rbf_pooled", offset].append(p_value(pooled))
                    # Two-sided: acceleration direction is not used to lower the p-value.
                    absolute = np.abs(summed_mean)
                    pvalues["paired_mean", offset].append(p_value(absolute))
            if (trial + 1) % 10 == 0:
                print(
                    model_name,
                    epoch_number,
                    "first-event",
                    case,
                    trial + 1,
                    trials,
                    flush=True,
                )
        for (variant, size), values in pvalues.items():
            rejects = sum(p <= 0.05 for p in values)
            excess = (
                binomtest(rejects, trials, 0.05, alternative="greater").pvalue
                if case == "H0"
                else None
            )
            report["results"].append(
                dict(
                    hypothesis=case,
                    variant=variant,
                    pairs=size,
                    rejections=rejects,
                    rejection_rate=rejects / trials,
                    wilson_95=wilson_interval(rejects, trials),
                    h0_excess_binomial_p=excess,
                    h0_excess_detected=(
                        bool(excess < 0.01 / 24) if excess is not None else None
                    ),
                    p_values=values,
                )
            )
        write_json(output / "report.json", report)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    report.update(status="completed", model_frozen_verified=True)
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=("FullyNN",), default="FullyNN")
    parser.add_argument("--epoch-number", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--seed", type=int, default=2026101419)
    run(**vars(parser.parse_args()))
