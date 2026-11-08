"""Burn sensitivity, encoding collisions and cost pilot; no final-power claim."""

import argparse
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import t as student

from new_ltpp.data.generation.convergent_hawkes import simulate, tail_mass


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def algebra():
    # Inclusion coefficients cancel exactly for every base RKHS feature map.
    p, q = [(1, 2), (3, 4)], [(1, 3), (2, 4)]
    coefficients = {
        i: sum(Fraction(i in x, 4) for x in p) - sum(Fraction(i in x, 4) for x in q)
        for i in range(1, 5)
    }
    assert all(v == 0 for v in coefficients.values())
    return dict(
        population_mmd_squared=0,
        exact_signed_feature_coefficients={
            str(k): str(v) for k, v in coefficients.items()
        },
        gaps_P=[1, 1],
        gaps_Q=[2, 2],
        scope="Linear inner product of mean base features only. Does not establish blindness of the library MKernel's nonlinear transformed distance.",
    )


def burn_pilot(plan, family, output):
    p = plan["p_values"][family]
    results = []
    for index, burn in enumerate(plan["burns"]):
        counts, rows = [], []
        for group in range(2):
            for unit in range(plan["independent_paths_per_group"]):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [plan["seed"], 1, family, index, group, unit]
                    )
                )
                visible, full = simulate(
                    rng,
                    burn=burn,
                    horizon=plan["horizon"],
                    p=p,
                    mu=plan["mu"],
                    b=plan["b"],
                    branching=plan["branching"],
                    max_events=plan["max_events_per_path"],
                )
                np.savez_compressed(
                    output / f"burn-{index}-g{group}-u{unit}.npz",
                    visible=visible,
                    full=full,
                )
                counts.append(len(visible))
                hist = np.histogram(visible, bins=np.linspace(0, 1, 11))[0]
                rows.append(
                    dict(
                        group=group,
                        unit=unit,
                        count=len(visible),
                        prehistory_count=int((full < 0).sum()),
                        mean_complete_gap=(
                            float(np.diff(visible).mean()) if len(visible) > 1 else None
                        ),
                        subwindow_counts=hist.tolist(),
                        visible_sha256=hashlib.sha256(visible.tobytes()).hexdigest(),
                    )
                )
        array = np.asarray(counts)
        margin = float(
            student.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array))
        )
        results.append(
            dict(
                burn=burn,
                independent_units=len(array),
                mean_count=float(array.mean()),
                mean_count_t95=[
                    float(array.mean() - margin),
                    float(array.mean() + margin),
                ],
                count_variance=float(array.var(ddof=1)),
                kernel_tail_mass=tail_mass(
                    burn, p=p, b=plan["b"], branching=plan["branching"]
                ),
                units=rows,
            )
        )
    return dict(
        status="complete_burn_pilot",
        p=p,
        results=results,
        interpretation=plan["stationarity"],
    )


def costs(plan, family, output):
    import torch

    from new_ltpp.configs.window_test_config import WindowTestConfig
    from new_ltpp.evaluation.statistical_testing.observations import from_configurations
    from new_ltpp.evaluation.statistical_testing.references import HawkesReference
    from scripts.ruche_trained_test_pilot import evaluate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, no CPU fallback")
    torch.set_num_threads(1)

    def pool(p, stream):
        rows = []
        for i in range(plan["full_pool_cost_n"]):
            rng = np.random.default_rng(
                np.random.SeedSequence([plan["seed"], 2, family, stream, i])
            )
            visible, full = simulate(
                rng,
                burn=plan["burns"][-1],
                p=p,
                mu=plan["mu"],
                b=plan["b"],
                branching=plan["branching"],
                max_events=plan["max_events_per_path"],
            )
            np.savez_compressed(
                output / f"g{stream}-u{i}.npz", visible=visible, full=full
            )
            rows.append(
                (torch.from_numpy(visible), torch.zeros(len(visible), dtype=torch.long))
            )
        return from_configurations(rows, device="cuda")

    x, y = pool(None, 0), pool(plan["p_values"][family], 1)
    methods = [dict(test_type="window_mmd", grid=g) for g in plan["grids"]]
    methods += [
        dict(test_type="window_mmd", representation=r) for r in ["events", "intervals"]
    ]
    methods += [
        dict(test_type="window_mmd", point_process_kernel="count_rbf", sigma=1.0),
        dict(test_type="window_mmd", space_kernel_type="linear"),
        dict(test_type="relative_loss"),
    ]
    ref = HawkesReference(rate=100, branching=0.5, beta=10, kind="exp")
    reports = []
    # A warmup of the actual public backend, separate from all measured methods.
    from scripts.check_backend import check_backend

    write(output / "backend.json", check_backend("cuda"))
    # These are solver-sensitivity controls on the same event paths, not new methods selected by power.
    from new_ltpp.evaluation.statistical_testing.observations import configurations

    tiny_x = from_configurations(configurations(x, 1, 1)[:2], device="cuda")
    tiny_y = from_configurations(configurations(y, 1, 1)[:2], device="cuda")
    for order in [0, 1, 2]:
        refinement = WindowTestConfig(
            test_type="window_mmd",
            representation="events",
            dyadic_order=order,
            horizon=1,
            num_event_types=1,
            n_samples=1,
            sigma=plan["sigma"],
        )
        write(
            output / f"event-refinement-{order}.json",
            evaluate(tiny_x, tiny_y, ref, [refinement], plan["max_comparison_seconds"]),
        )
    for index, method in enumerate(methods):
        cfg = WindowTestConfig(
            horizon=1,
            num_event_types=1,
            n_samples=plan["permutations"],
            seed=plan["seed"],
            **({"sigma": plan["sigma"]} | method),
        )
        started = time.perf_counter()
        try:
            report = evaluate(x, y, ref, [cfg], plan["max_comparison_seconds"])
            # P-values here only measure execution; no power estimate or tuning.
            report["status"] = "cost_probe_complete_not_power"
        except Exception as error:
            report = dict(
                status="blocked_method",
                config=cfg.model_dump(),
                error_type=type(error).__name__,
                error=str(error),
                elapsed_seconds=time.perf_counter() - started,
            )
        write(output / f"method-{index}.json", report)
        reports.append(dict(method=index, status=report["status"]))
    return dict(
        status="complete_cost_pilot_with_all_failures_retained",
        methods=reports,
        warning="One draw/case is a cost probe, never calibration or power. Methods/burn may only be chosen for fresh independent future evaluation.",
    )


def evaluation(plan, family, output, pilot_root):
    """Fresh-bank finite-initialization study, gated by every cost result."""
    import torch

    from new_ltpp.configs.window_test_config import WindowTestConfig
    from new_ltpp.evaluation.statistical_testing.observations import from_configurations
    from new_ltpp.evaluation.statistical_testing.references import HawkesReference
    from scripts.ruche_trained_test_pilot import evaluate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, no CPU fallback")
    torch.set_num_threads(1)
    cost = pilot_root / f"cost-{family}"
    summaries = [json.loads((cost / f"method-{i}.json").read_text()) for i in range(10)]
    if any(r.get("status") != "cost_probe_complete_not_power" for r in summaries):
        raise RuntimeError(
            "A prespecified method failed its cost/numerical gate; no method excluded"
        )
    if sum(r["full_pool_seconds"] for r in summaries) > plan["max_comparison_seconds"]:
        raise RuntimeError("Full suite exceeds the prespecified 60-second cost gate")
    burn = json.loads((pilot_root / f"burn-{family}" / "summary.json").read_text())
    if burn["status"] != "complete_burn_pilot":
        raise RuntimeError("Burn-in sensitivity evidence incomplete")
    # B=500 is fixed before these fresh draws. Retain finite-start interpretation
    # regardless of the sensitivity study; never label it exact stationarity.
    reference = HawkesReference(rate=100, branching=0.5, beta=10, kind="exp")
    with (output / "trials.jsonl").open("x") as target:
        for trial in range(plan["evaluation_trials"]):
            pairs = []
            for group, p in enumerate([None, plan["p_values"][family]]):
                rows = []
                times_all = []
                full_hashes = []
                for unit in range(plan["full_pool_cost_n"]):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [plan["seed"], 3, family, trial, group, unit]
                        )
                    )
                    visible, full = simulate(
                        rng,
                        burn=plan["burns"][-1],
                        p=p,
                        mu=plan["mu"],
                        b=plan["b"],
                        branching=plan["branching"],
                        max_events=plan["max_events_per_path"],
                    )
                    rows.append(
                        (
                            torch.from_numpy(visible),
                            torch.zeros(len(visible), dtype=torch.long),
                        )
                    )
                    times_all.append(visible)
                    full_hashes.append(hashlib.sha256(full.tobytes()).hexdigest())
                data = from_configurations(rows, device="cuda")
                np.savez_compressed(
                    output / f"trial-{trial:03d}-g{group}.npz",
                    times=data.time_seqs.cpu().numpy(),
                    marks=data.type_seqs.cpu().numpy(),
                    mask=data.valid_event_mask.cpu().numpy(),
                    full_history_sha256=np.asarray(full_hashes),
                    horizon=np.array(1.0),
                )
                pairs.append(data)
            seed = int(
                np.random.SeedSequence(
                    [plan["seed"], 3, family, trial, 3]
                ).generate_state(1)[0]
            )
            settings = [
                WindowTestConfig.model_validate(
                    r["tests"][0]["config"] | {"seed": seed}
                )
                for r in summaries
            ]
            report = evaluate(
                *pairs, reference, settings, plan["max_comparison_seconds"]
            )
            report.update(
                trial=trial,
                family=family,
                p=plan["p_values"][family],
                burn=plan["burns"][-1],
            )
            target.write(json.dumps(report, allow_nan=False) + "\n")
            target.flush()
    return dict(
        status="complete_finite_start_pilot",
        family=family,
        p=plan["p_values"][family],
        n=128,
        trials=plan["evaluation_trials"],
        burn=plan["burns"][-1],
        methods=10,
        interpretation="Finite-start B=500, not exact stationarity. 100 independent trials/case is a pilot, no hyperparameter selection on evaluation. Read burn-sensitivity report alongside powers.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["burn", "cost", "algebra", "evaluation"], required=True
    )
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--family", type=int, choices=range(8), default=0)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(a.plan.read_text())
    write(
        a.output / "provenance.json",
        dict(
            plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
            phase=a.phase,
            numpy=np.__version__,
            family=a.family,
            seed=plan["seed"],
        ),
    )
    try:
        result = (
            algebra()
            if a.phase == "algebra"
            else (
                burn_pilot(plan, a.family, a.output)
                if a.phase == "burn"
                else (
                    costs(plan, a.family, a.output)
                    if a.phase == "cost"
                    else evaluation(plan, a.family, a.output, a.pilot_root)
                )
            )
        )
    except Exception as error:
        write(
            a.output / "summary.json",
            dict(status="failed", error_type=type(error).__name__, error=str(error)),
        )
        raise
    write(a.output / "summary.json", result)
    print(json.dumps(result), flush=True)
