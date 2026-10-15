"""Opt-in RMSE replay on the report campaign's exact observations and swaps.

Descriptive completed-gap RMSE and a separate fixed-landmark, censored-target
loss test. No fitting, simulator changes or production-model changes.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
import torch
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.stats import beta, binomtest

from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)
from scripts.report_experiment_summary import holm_adjust
from scripts.report_experiment_validation import (
    CASES,
    METHODS,
    ROOT,
    event_hash,
    read_records,
    seed_entropy,
    simulate_events,
    uniformize_counts,
    write_json,
)

PROTOCOL = "common_reference_rmse_v1"
LANDMARKS = (0.0, 0.25, 0.5, 0.75)
CAP = 0.25


@lru_cache(maxsize=8)
def quadrature(order):
    return leggauss(order)


def cumulative_hazard(delays, history, time, process):
    """Conditional hazard before the next arrival, including the current event."""
    delays = np.asarray(delays, dtype=np.float64)
    if process.kind not in ("poisson", "exp", "power"):
        raise ValueError("Unknown predictor")
    if process.kind == "poisson":
        return process.rate * delays
    mu = process.rate * (1 - process.branching)
    ages = time - np.asarray(history, dtype=np.float64)
    if np.any(ages < 0):
        raise ValueError("Prediction must not see future events")
    if process.kind == "exp":
        excitation = process.branching * np.exp(-process.beta * ages).sum()
        return mu * delays + excitation * (-np.expm1(-process.beta * delays))
    base = 1 + process.beta * ages
    extra = process.branching * (
        base[:, None] ** (1 - process.exponent)
        - (base[:, None] + process.beta * delays.reshape(1, -1))
        ** (1 - process.exponent)
    ).sum(0)
    return mu * delays + extra.reshape(delays.shape)


def waiting_mean(history, time, process, cap=None, order=64):
    """E[min(W, cap)|visible history], or E[W] for descriptive gap RMSE.

    The pre-window past is unavailable, so this is a frozen prediction rule,
    not an oracle claim for Hawkes simulated with burn-in. Two quadrature orders
    are compared; adaptive integration verifies any disagreement, which fails.
    """
    if cap is not None and (not np.isfinite(cap) or cap <= 0):
        raise ValueError("Positive finite observation cap required")
    if process.kind == "poisson":
        return (
            1 / process.rate
            if cap is None
            else -np.expm1(-process.rate * cap) / process.rate
        )

    def survivor(s):
        return float(
            np.exp(-cumulative_hazard(np.array([s]), history, time, process)[0])
        )

    if cap is None:
        value, error = quad(survivor, 0, np.inf, epsabs=1e-10, epsrel=1e-8)
        if error > 1e-8 * max(value, 1e-2):
            raise ArithmeticError("Unresolved waiting-time integral")
        return value
    nodes, weights = quadrature(order)
    s = (nodes + 1) * cap / 2
    value = float(
        weights @ np.exp(-cumulative_hazard(s, history, time, process)) * cap / 2
    )
    # Cheap, independent-order check on every prediction, no silent clipping.
    nodes2, weights2 = quadrature(order // 2)
    s2 = (nodes2 + 1) * cap / 2
    check = float(
        weights2 @ np.exp(-cumulative_hazard(s2, history, time, process)) * cap / 2
    )
    if not np.isclose(value, check, rtol=1e-8, atol=1e-11):
        check, error = quad(survivor, 0, cap, epsabs=1e-11, epsrel=1e-9)
        if error > 1e-9 * max(check, 1e-2) or not np.isclose(
            value, check, rtol=1e-8, atol=1e-11
        ):
            raise ArithmeticError("Unresolved restricted waiting-time integral")
    if not 0 < value <= cap:
        raise ArithmeticError("Invalid restricted mean")
    return value


def check_rows(rows, horizon):
    if horizon != 1.0 or not rows:
        raise ValueError(
            "This prespecified protocol requires horizon=1 and nonempty bank"
        )
    for row in rows:
        if (
            not np.isfinite(row).all()
            or np.any(np.diff(row) < 0)
            or np.any(row < 0)
            or np.any(row > horizon)
        ):
            raise ValueError("Invalid events")


def landmark_losses(rows, reference, predictor=None, horizon=1.0):
    """Equal weight per path and fixed landmark, including paths with no events.

    Predict a restricted waiting time, NOT the unobserved event beyond the cap.
    The denominator is the H0 reference prediction, identical across candidates
    for each history. It does not depend on future outcomes or fitted data.
    """
    check_rows(rows, horizon)
    predictor = reference if predictor is None else predictor
    losses = []
    for row in rows:
        errors = []
        for time in LANDMARKS:
            pos = int(np.searchsorted(row, time, side="right"))
            history = row[:pos]
            target = min(float(row[pos] - time), CAP) if pos < len(row) else CAP
            scale = waiting_mean(history, time, reference, CAP)
            prediction = (
                scale
                if predictor == reference
                else waiting_mean(history, time, predictor, CAP)
            )
            errors.append(((target - prediction) / scale) ** 2)
        losses.append(float(np.mean(errors)))
    return np.asarray(losses)


def gap_rmse(rows, reference, predictor):
    """Descriptive completed-gap RMSE; terminal censored gaps are not observed.

    All candidates share each event's reference denominator. Unlike the archived
    exponential recurrence, history includes the event at the forecast origin.
    Empty/singleton paths are reported as unavailable, never zero error.
    """
    check_rows(rows, 1.0)
    path_mse = []
    for row in rows:
        if len(row) < 2:
            continue
        errors = []
        for i in range(len(row) - 1):
            history, time = row[: i + 1], float(row[i])
            scale = waiting_mean(history, time, reference)
            prediction = (
                scale
                if predictor == reference
                else waiting_mean(history, time, predictor)
            )
            errors.append(((row[i + 1] - time - prediction) / scale) ** 2)
        path_mse.append(float(np.mean(errors)))
    return dict(
        rmse=float(np.sqrt(np.mean(path_mse))) if path_mse else None,
        scored_paths=len(path_mse),
        excluded_paths=len(rows) - len(path_mse),
        completed_gaps=sum(max(0, len(row) - 1) for row in rows),
    )


def prediction_curves(source_root, output, *, completed_gaps=False):
    """The four historical parameter sweeps, with matched T=1 and n=128.

    All candidates in a sweep score the same first archived X bank; no new data
    are selected in response to the ranking. These are descriptive curves, not
    power estimates. Full completed-gap curves are an explicitly optional cost.
    """
    from scripts.report_experiment_validation import EXP, POISSON, POWER, Process

    groups = [
        (
            "poisson",
            "poisson_H0",
            POISSON,
            [
                (str(r), Process("poisson", rate=100 * r))
                for r in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0)
            ],
        ),
        (
            "hawkes_opt",
            "hawkes_power_H0",
            POWER,
            [(str(a), Process("exp", beta=2 * a)) for a in range(2, 9)],
        ),
        (
            "hawkes_kernel",
            "hawkes_exp_H0",
            EXP,
            [(str(p), Process("power", exponent=p)) for p in (1.5, 2.0, 3.0, 5.0, 8.0)],
        ),
        (
            "hawkes_improved",
            "hawkes_exp_H0",
            EXP,
            [(str(a), Process("exp", branching=a / 10)) for a in range(2, 9)],
        ),
    ]
    results = []
    for name, case, reference, candidates in groups:
        source = Path(source_root) / case / "cohort-0"
        plan = json.loads((source / "plan.json").read_text())
        with (source / "trials.jsonl").open() as handle:
            first = json.loads(handle.readline())
        if (
            first["trial"] != 0
            or first["status"] != "completed"
            or plan["n"] != 128
            or plan["horizon"] != 1
            or plan["evaluation_domain"] != 0
        ):
            raise ValueError("Curves require the prespecified first primary bank")
        rows, _ = regenerate(plan, first)
        points = []
        for label, predictor in [
            ("reference", reference),
            ("constant_rate_100", POISSON),
            *candidates,
        ]:
            point = dict(
                parameter=label,
                predictor=asdict(predictor),
                restricted_landmark_rmse=float(
                    np.sqrt(np.mean(landmark_losses(rows, reference, predictor)))
                ),
            )
            if completed_gaps:
                point["completed_gap"] = gap_rmse(rows, reference, predictor)
            points.append(point)
            print(name, label, point["restricted_landmark_rmse"], flush=True)
        results.append(
            dict(
                experiment=name,
                source_case=case,
                reference=asdict(reference),
                source_commit=plan["commit"],
                x_sha256=first["x_sha256"],
                n=plan["n"],
                horizon=plan["horizon"],
                points=points,
            )
        )
        write_json(
            output,
            dict(
                protocol=PROTOCOL,
                status="completed" if len(results) == len(groups) else "in_progress",
                rmse_source_sha256=hashlib.sha256(
                    Path(__file__).read_text().encode()
                ).hexdigest(),
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                completed_gaps=completed_gaps,
                descriptive_only=True,
                curves=results,
            ),
        )
    return results


def loss_test(x_loss, y_loss, swap_seed, permutations):
    """Two-sided mean-MSE difference; paired swaps match the MMD exactly."""
    x_loss, y_loss = np.asarray(x_loss), np.asarray(y_loss)
    if (
        x_loss.shape != y_loss.shape
        or x_loss.ndim != 1
        or len(x_loss) < 2
        or not np.isfinite(x_loss + y_loss).all()
    ):
        raise ValueError("Finite equally sized independent pairs required")
    generator = torch.Generator().manual_seed(swap_seed)
    signs = torch.cat(
        (
            torch.ones(1, len(x_loss)),
            1 - 2 * torch.randint(2, (permutations, len(x_loss)), generator=generator),
        )
    ).double()
    values = (signs @ torch.from_numpy(x_loss - y_loss).double() / len(x_loss)).square()
    return dict(
        p_value=float(monte_carlo_p_value(values[0], values[1:])),
        observed=float(values[0]),
        null=values[1:].tolist(),
        rmse_x=float(np.sqrt(np.mean(x_loss))),
        rmse_y=float(np.sqrt(np.mean(y_loss))),
        path_mse_x=x_loss.tolist(),
        path_mse_y=y_loss.tolist(),
    )


def regenerate(plan, row):
    case, cohort, trial = plan["case"], plan["cohort"], row["trial"]
    domain = plan["evaluation_domain"]
    x = simulate_events(
        np.random.default_rng(
            np.random.SeedSequence(seed_entropy(case, cohort, trial, 1, domain))
        ),
        CASES[case][0],
        plan["n"],
        plan["horizon"],
        plan["burn"],
    )
    ry = np.random.default_rng(
        np.random.SeedSequence(seed_entropy(case, cohort, trial, 2, domain))
    )
    y = (
        uniformize_counts(x, ry, plan["horizon"])
        if CASES[case][2]
        else simulate_events(
            ry, CASES[case][1], plan["n"], plan["horizon"], plan["burn"]
        )
    )
    seed = int(
        np.random.SeedSequence(
            seed_entropy(case, cohort, trial, 3, domain)
        ).generate_state(1, dtype=np.uint64)[0]
    ) % (2**63)
    if (
        event_hash(x) != row["x_sha256"]
        or event_hash(y) != row["y_sha256"]
        or seed != row["swap_seed"]
    ):
        raise ValueError("Replay does not match archived MMD observations/swaps")
    return x, y


def replay(source, output, *, descriptive=False):
    source, output = Path(source), Path(output)
    plan = json.loads((source / "plan.json").read_text())
    raw = (source / "trials.jsonl").read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Source still being appended; wait for complete MMD cell")
    records = [json.loads(line) for line in raw.splitlines()]
    if (
        len(records) != plan["trials"]
        or [r["trial"] for r in records] != list(range(plan["trials"]))
        or any(r["status"] != "completed" for r in records)
    ):
        raise ValueError("All planned MMD trials must succeed; no filtered subset")
    design = dict(
        protocol=PROTOCOL,
        source_plan=plan,
        source_directory=str(source.resolve()),
        source_raw_sha256=hashlib.sha256(raw).hexdigest(),
        rmse_source_sha256=hashlib.sha256(
            Path(__file__).read_text().encode()
        ).hexdigest(),
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_committed=subprocess.run(
            ["git", "cat-file", "-e", "HEAD:scripts/report_rmse_validation.py"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0,
        tracked_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        numpy=np.__version__,
        scipy=scipy.__version__,
        torch=torch.__version__,
        landmarks=list(LANDMARKS),
        cap=CAP,
        descriptive=descriptive,
        reference=asdict(CASES[plan["case"]][0]),
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists() and json.loads(
        (output / "plan.json").read_text()
    ) != design:
        raise ValueError("Changed source or RMSE protocol on resume")
    write_json(output / "plan.json", design)
    done = read_records(output / "trials.jsonl")
    if len(done) > len(records):
        raise ValueError("RMSE checkpoint exceeds source")
    for record in records[len(done) :]:
        result = dict(trial=record["trial"], status="in_progress")
        try:
            x, y = regenerate(plan, record)
            ref, candidate = CASES[plan["case"]][:2]
            test = loss_test(
                landmark_losses(x, ref),
                landmark_losses(y, ref),
                record["swap_seed"],
                plan["permutations"],
            )
            result.update(
                status="completed",
                rmse=test,
                mmd=record["methods"],
                x_sha256=record["x_sha256"],
                y_sha256=record["y_sha256"],
            )
            # Every trial compares candidate predictions on the SAME X histories,
            # using the H0 reference denominator. No in-sample mean baseline.
            result["prediction_rmse_on_x"] = dict(
                reference=float(np.sqrt(np.mean(test["path_mse_x"]))),
                candidate=float(np.sqrt(np.mean(landmark_losses(x, ref, candidate)))),
            )
            if descriptive and record["trial"] == 0:
                result["completed_gap_diagnostic"] = dict(
                    reference=gap_rmse(x, ref, ref),
                    candidate=gap_rmse(x, ref, candidate),
                )
        except Exception as error:
            result.update(status="failed", error=repr(error))
        with (output / "trials.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(result, allow_nan=False) + "\n")
        done.append(result)
        if len(done) % 25 == 0:
            print(
                plan["case"], plan["cohort"], len(done), "/", len(records), flush=True
            )
    errors = [r for r in done if r["status"] != "completed"]
    report = dict(
        design=design,
        status="completed" if not errors else "completed_with_errors",
        errors=errors,
        trials=len(done),
    )
    write_json(output / "report.json", report)
    if errors:
        raise RuntimeError("RMSE replay contains failed trials")
    return report


def aggregate(root):
    """Strict 20-cell paired comparison; no optional stopping or filtered trials."""
    root = Path(root)
    errors, calibration, power, predictions = [], [], [], []
    provenance, source_provenance = None, None
    for case in CASES:
        pooled = []
        for cohort in (0, 1):
            try:
                directory = root / case / f"cohort-{cohort}"
                report = json.loads((directory / "report.json").read_text())
                design = report["design"]
                if design != json.loads((directory / "plan.json").read_text()):
                    raise ValueError("Report and plan disagree")
                signature = {
                    k: design[k]
                    for k in (
                        "protocol",
                        "rmse_source_sha256",
                        "commit",
                        "numpy",
                        "scipy",
                        "torch",
                        "landmarks",
                        "cap",
                        "descriptive",
                        "source_committed",
                        "tracked_dirty",
                    )
                }
                if (
                    signature["protocol"] != PROTOCOL
                    or signature["landmarks"] != list(LANDMARKS)
                    or signature["cap"] != CAP
                ):
                    raise ValueError("Unrecognized RMSE protocol")
                if not signature["source_committed"] or signature["tracked_dirty"]:
                    raise ValueError("Primary RMSE source must be committed and clean")
                if provenance is not None and signature != provenance:
                    raise ValueError("Mixed RMSE source/environment")
                provenance = signature
                plan = design["source_plan"]
                source_sig = {
                    k: plan[k]
                    for k in (
                        "commit",
                        "source_lf_sha256",
                        "lock_sha256",
                        "numpy",
                        "torch",
                        "pysiglib",
                        "device",
                        "methods",
                        "master_seed",
                    )
                }
                if source_provenance is not None and source_sig != source_provenance:
                    raise ValueError("Mixed original MMD campaigns")
                source_provenance = source_sig
                expected = dict(
                    case=case,
                    cohort=cohort,
                    evaluation_domain=0,
                    n=128,
                    permutations=499,
                    horizon=1.0,
                    burn=10.0,
                    grid=100,
                    trials=1000 if case.endswith("H0") else 250,
                )
                if (
                    any(plan.get(k) != v for k, v in expected.items())
                    or not plan["source_committed"]
                    or plan["tracked_dirty"]
                ):
                    raise ValueError("Not the primary scientific design")
                records = read_records(directory / "trials.jsonl")
                if (
                    report["status"] != "completed"
                    or report["errors"]
                    or len(records) != expected["trials"]
                    or any(r["status"] != "completed" for r in records)
                ):
                    raise ValueError("Incomplete/failed cell")
                original = Path(design["source_directory"])
                raw = (original / "trials.jsonl").read_bytes()
                if (
                    hashlib.sha256(raw).hexdigest() != design["source_raw_sha256"]
                    or json.loads((original / "plan.json").read_text()) != plan
                ):
                    raise ValueError("Original MMD source changed after replay")
                originals = [json.loads(line) for line in raw.splitlines()]
                if len(originals) != len(records):
                    raise ValueError("Original/replay trial counts disagree")
                for row in records:
                    origin = originals[row["trial"]]
                    if (
                        origin["methods"] != row["mmd"]
                        or origin["x_sha256"] != row["x_sha256"]
                        or origin["y_sha256"] != row["y_sha256"]
                    ):
                        raise ValueError("Paired MMD draws or data identifiers changed")
                    seed = int(
                        np.random.SeedSequence(
                            seed_entropy(case, cohort, row["trial"], 3)
                        ).generate_state(1, dtype=np.uint64)[0]
                    ) % (2**63)
                    actual = loss_test(
                        row["rmse"]["path_mse_x"], row["rmse"]["path_mse_y"], seed, 499
                    )
                    if actual != row["rmse"]:
                        raise ValueError("RMSE summary differs from archived losses")
                    for method in METHODS:
                        draw = row["mmd"][method]
                        p = float(
                            monte_carlo_p_value(
                                torch.tensor(draw["observed"], dtype=torch.float64),
                                torch.tensor(draw["null"], dtype=torch.float64),
                            )
                        )
                        if p != draw["p_value"]:
                            raise ValueError("MMD summary differs from archived swaps")
                if case.endswith("H0"):
                    calibration.extend(
                        calibration_row(case, cohort, records, method)
                        for method in ("rmse", *METHODS)
                    )
                pooled.extend(records)
            except Exception as error:
                errors.append(dict(case=case, cohort=cohort, error=repr(error)))
        if len(pooled) != (2000 if case.endswith("H0") else 500):
            continue
        if case.endswith("H0"):
            calibration.extend(
                calibration_row(case, "pooled", pooled, method)
                for method in ("rmse", *METHODS)
            )
        else:
            for method in METHODS:
                a = sum(
                    r["rmse"]["p_value"] <= 0.05 and r["mmd"][method]["p_value"] > 0.05
                    for r in pooled
                )
                b = sum(
                    r["rmse"]["p_value"] > 0.05 and r["mmd"][method]["p_value"] <= 0.05
                    for r in pooled
                )
                power.append(
                    dict(
                        case=case,
                        mmd_method=method,
                        rmse_only=a,
                        mmd_only=b,
                        rmse=rate_row([r["rmse"]["p_value"] for r in pooled]),
                        mmd=rate_row([r["mmd"][method]["p_value"] for r in pooled]),
                        paired_p=float(binomtest(a, a + b).pvalue) if a + b else 1.0,
                    )
                )
        predictions.append(
            dict(
                case=case,
                trials=len(pooled),
                reference_rmse=float(
                    np.sqrt(
                        np.mean(
                            [
                                r["prediction_rmse_on_x"]["reference"] ** 2
                                for r in pooled
                            ]
                        )
                    )
                ),
                candidate_rmse=float(
                    np.sqrt(
                        np.mean(
                            [
                                r["prediction_rmse_on_x"]["candidate"] ** 2
                                for r in pooled
                            ]
                        )
                    )
                ),
            )
        )
    for row, p in zip(
        power, holm_adjust([r["paired_p"] for r in power], family_size=18)
    ):
        row["holm_p_over_18_comparisons"] = p
    complete = not errors and len(calibration) == 48 and len(power) == 18
    result = dict(
        status="complete" if complete else "incomplete",
        errors=errors,
        provenance=provenance,
        source_provenance=source_provenance,
        calibration=calibration,
        paired_power=power,
        prediction_rmse=predictions,
    )
    write_json(root / "summary.json", result)
    if not complete or any(r["h0_excess_detected"] for r in calibration):
        raise RuntimeError("Campaign incomplete or H0 excess; see summary.json")
    return result


def rate_row(ps):
    k, n = sum(p <= 0.05 for p in ps), len(ps)
    ci = binomtest(k, n).proportion_ci()
    return dict(rejected=k, trials=n, rate=k / n, ci95=[ci.low, ci.high])


def calibration_row(case, cohort, records, method="rmse"):
    row = rate_row(
        [
            r["rmse"]["p_value"] if method == "rmse" else r["mmd"][method]["p_value"]
            for r in records
        ]
    )
    p = float(
        binomtest(row["rejected"], row["trials"], 0.05, alternative="greater").pvalue
    )
    # Across original MMD (36) + this addition (12), a conservative family of 48.
    row.update(
        case=case,
        cohort=cohort,
        method=method,
        excess_p=p,
        h0_excess_detected=p < 0.01 / 48,
    )
    if cohort == "pooled":
        k, n = row["rejected"], row["trials"]
        row["simultaneous_upper_99_over_16"] = (
            float(beta.ppf(1 - 0.01 / 16, k + 1, n - k)) if k < n else 1.0
        )
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    cell = subs.add_parser("replay")
    cell.add_argument("source", type=Path)
    cell.add_argument("output", type=Path)
    cell.add_argument("--descriptive", action="store_true")
    summary = subs.add_parser("aggregate")
    summary.add_argument("root", type=Path)
    curves = subs.add_parser("curves")
    curves.add_argument("source", type=Path)
    curves.add_argument("output", type=Path)
    curves.add_argument("--completed-gaps", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "replay":
        replay(args.source, args.output, descriptive=args.descriptive)
    elif args.command == "aggregate":
        aggregate(args.root)
    else:
        prediction_curves(args.source, args.output, completed_gaps=args.completed_gaps)
