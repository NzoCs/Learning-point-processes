"""Independent finite-window calibration for frozen FullyNN/RMTPP checkpoints."""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import binomtest

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import (
    cached_paired_statistics,
    heterogeneous_histories,
    swap_orders,
)
from scripts.extended_scientific_validation import code_provenance
from scripts.integration_suite import write_json
from scripts.scientific_calibration import wilson_interval
from scripts.stopped_conditional_calibration import (
    encoded_last_state,
    load_frozen_model,
    numpy_hazard,
)


def window_continuations(
    model, histories, horizon, *, rng=None, speed=1.0, max_events=1024
):
    """Observe until the fixed time, not until an event quota.

    rng=None invokes the actual production one-step simulation. Otherwise the
    decoder/inversion/mark RNG are independently implemented in NumPy/SciPy.
    An exhausted computation budget raises; it never declares a completed window.
    """
    n, initial = histories.time_seqs.shape
    times = torch.zeros(n, initial + max_events, dtype=torch.float32)
    marks = torch.full(times.shape, model.pad_token_id, dtype=torch.long)
    times[:, :initial], marks[:, :initial] = histories.time_seqs, histories.type_seqs
    origin = histories.time_seqs[:, -1].double()
    clock = origin.clone()
    alive = torch.ones(n, dtype=torch.bool)
    paths = [[] for _ in range(n)]
    for step in range(max_events):
        rows = alive.nonzero(as_tuple=True)[0]
        if not len(rows):
            break
        prefix_t, prefix_m = (
            times[rows, : initial + step],
            marks[rows, : initial + step],
        )
        prefix = Batch(
            prefix_t,
            torch.diff(prefix_t, prepend=torch.zeros(len(rows), 1)),
            prefix_m,
            torch.ones_like(prefix_t, dtype=torch.bool),
        )
        if rng is None:
            with torch.no_grad():
                delay, mark = model._simulator._simulate_one_step(**prefix.to_mapping())
            delay, mark = delay.flatten().double(), mark.flatten()
        else:
            delays, next_marks = [], []
            for hidden in encoded_last_state(model, prefix):
                hazard, marked, limit = numpy_hazard(model, hidden)
                target = rng.exponential()
                if target >= limit:
                    delays.append(math.inf)
                    next_marks.append(model.pad_token_id)
                    continue
                # Only invert if the event belongs inside the remaining window.
                row = rows[len(delays)]
                remaining = float(horizon - (clock[row] - origin[row]))
                if hazard(remaining * speed) < target:
                    delays.append(math.inf)
                    next_marks.append(model.pad_token_id)
                    continue
                dt = brentq(
                    lambda t: hazard(t) - target, 0.0, remaining * speed, xtol=1e-12
                )
                intensity = marked(dt)
                delays.append(dt / speed)
                next_marks.append(
                    rng.choice(model.num_event_types, p=intensity / intensity.sum())
                )
            delay, mark = (
                torch.tensor(delays, dtype=torch.double),
                torch.tensor(next_marks),
            )
        proposed = clock[rows] + delay
        if torch.isnan(proposed).any() or (delay <= 0).any():
            raise RuntimeError("Invalid window simulation proposal")
        inside = torch.isfinite(proposed) & (proposed - origin[rows] <= horizon)
        alive[rows[~inside]] = False
        selected = rows[inside]
        if (
            (proposed[inside].float() - times[selected, initial + step - 1]) <= 0
        ).any():
            raise RuntimeError("Window clock stagnation")
        times[selected, initial + step], marks[selected, initial + step] = (
            proposed[inside].float(),
            mark[inside],
        )
        clock[selected] = proposed[inside]
        for row, value, kind in zip(
            selected.tolist(),
            (clock[selected] - origin[selected]).tolist(),
            mark[inside].tolist(),
        ):
            paths[row].append((value, kind))
    if alive.any():
        raise RuntimeError("Window simulation event safety budget exhausted")
    length = max(1, max(map(len, paths)))
    t = torch.zeros(n, length)
    m = torch.full((n, length), model.pad_token_id, dtype=torch.long)
    mask = torch.zeros(n, length, dtype=torch.bool)
    for row, path in enumerate(paths):
        for col, (value, kind) in enumerate(path):
            t[row, col], m[row, col], mask[row, col] = value, kind, True
    gaps = torch.diff(t, prepend=torch.zeros(n, 1)).masked_fill(~mask, 0)
    return Batch(t, gaps, m, mask)  # Window censoring carries no permanent-end flag.


def run(
    name,
    output,
    *,
    checkpoint=None,
    trials=300,
    power_trials=100,
    permutations=99,
    horizon=2.0,
    seed=20261101,
):
    if (
        min(trials, power_trials, permutations) < 1
        or not math.isfinite(horizon)
        or horizon <= 0
    ):
        raise ValueError("Positive trial counts and finite positive horizon required")
    torch.set_num_threads(1)
    if (output / "report.json").exists():
        raise FileExistsError("Use a fresh calibration output directory")
    model, fit = load_frozen_model(name, output, checkpoint)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    report = dict(
        model=name,
        code=code_provenance(__file__),
        fit=fit,
        seed=seed,
        horizon=horizon,
        trials_H0=trials,
        power_trials=power_trials,
        permutations=permutations,
        batch_sizes=[16, 11, 5],
        results=[],
        status="in_progress",
        scope="Independent NumPy/SciPy decoder and RNG, shared frozen Torch encoder; fixed time-window censoring, no terminal flag. No fitting uncertainty or CUDA validation.",
        acceptance="One-sided H0 binomial excess test at 0.01/16 across two models, two checkpoints, two designs and two aggregation methods; no tuning",
    )
    for case_id, case in enumerate(("H0", "twice_speed", "quiet_window")):
        values = {"single": [], "aggregated": []}
        digest = hashlib.sha256()
        count = trials if case == "H0" else power_trials
        for trial in range(count):
            test = ConditionalMMDTwoSampleTest(
                SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2),
                permutations,
                observation_horizon=horizon,
            )
            for block, n in enumerate(report["batch_sizes"]):
                h = heterogeneous_histories(n)
                sample_seed = seed + case_id * 10000000 + trial * 1000 + block * 10
                torch.manual_seed(sample_seed)
                y = window_continuations(model, h, horizon)
                x = window_continuations(
                    model,
                    h,
                    horizon,
                    rng=np.random.default_rng(sample_seed + 100000000),
                    speed=2.0 if case == "twice_speed" else 1.0,
                )
                if case == "quiet_window":
                    x.valid_event_mask.zero_()
                    x.time_seqs.zero_()
                    x.time_delta_seqs.zero_()
                for b in (x, y):
                    digest.update(b.time_seqs.numpy().tobytes())
                    digest.update(b.type_seqs.numpy().tobytes())
                    digest.update(b.valid_event_mask.numpy().tobytes())
                orders = swap_orders(n, permutations, sample_seed + 200000000)
                stats = cached_paired_statistics(test, h, x, y, orders)
                if trial == 0:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(sample_seed + 200000000)
                        actual = test.compute_statistics(
                            x, y, histories=h, accumulate=False
                        )
                    torch.testing.assert_close(
                        stats[0], actual["observed_statistic"], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        stats[1:], actual["permuted_statistics"], rtol=0, atol=0
                    )
                test._accumulate(stats[0], stats[1:])
                if block == 0:
                    values["single"].append(
                        float(monte_carlo_p_value(stats[0], stats[1:]))
                    )
            values["aggregated"].append(float(test.get_final_p_value()))
            if (trial + 1) % 50 == 0:
                print(name, "window", case, trial + 1, count, flush=True)
        for method, pvalues in values.items():
            rejected = sum(p <= 0.05 for p in pvalues)
            excess = (
                binomtest(rejected, count, 0.05, alternative="greater").pvalue
                if case == "H0"
                else None
            )
            report["results"].append(
                dict(
                    hypothesis=case,
                    method=method,
                    rejections=rejected,
                    trials=count,
                    rejection_rate=rejected / count,
                    wilson_95=wilson_interval(rejected, count),
                    excess_binomial_p=excess,
                    h0_excess_detected=(
                        bool(excess < 0.01 / 16) if excess is not None else None
                    ),
                    p_values=pvalues,
                    data_sha256=digest.hexdigest(),
                    public_api_verified_on_first_trial=True,
                )
            )
        write_json(output / "report.json", report)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
    report.update(status="completed", model_frozen_verified=True)
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("FullyNN", "RMTPP"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--power-trials", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261101)
    args = vars(parser.parse_args())
    args["name"] = args.pop("model")
    run(**args)
