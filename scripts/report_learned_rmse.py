"""Learn on separate H0 banks, freeze, replay matched all-event RMSE/MMD."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import binomtest

from scripts.all_event_rmse import all_event_losses
from scripts.learned_rmse_reference import fit_reference
from scripts.report_experiment_summary import holm_adjust
from scripts.report_experiment_validation import (
    CASES,
    METHODS,
    PILOT_LAWS,
    ROOT,
    Process,
    event_hash,
    read_records,
    simulate_events,
    write_json,
)
from scripts.report_rmse_validation import (
    calibration_row,
    loss_test,
    rate_row,
    regenerate,
)

PROTOCOL = "learned_common_reference_all_event_rmse_v1"
TRAIN_SEED = 2026101931
TRAIN_N = 1024
COMPARATORS = (*METHODS, "known_reference")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_records(path):
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Incomplete frozen archive; refusing to modify it")
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(r["trial"] != i or r["status"] != "completed" for i, r in enumerate(rows)):
        raise ValueError("Frozen archive has failed or noncontiguous trials")
    return rows


def provenance():
    return dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
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
        sources={
            name: digest(ROOT / "scripts" / name)
            for name in (
                "report_learned_rmse.py",
                "learned_rmse_reference.py",
                "all_event_rmse.py",
                "report_rmse_validation.py",
            )
        },
    )


def train(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training-plan.json").exists():
        raise ValueError(
            "Training is non-resumable: preserve this bank and use a new directory"
        )
    design = dict(
        protocol=PROTOCOL,
        master_seed=TRAIN_SEED,
        training_paths=TRAIN_N,
        cohorts=2,
        families=[p.kind for p in PILOT_LAWS],
        horizon=1.0,
        burn=10.0,
        provenance=provenance(),
    )
    if design["provenance"]["tracked_dirty"]:
        raise ValueError("Training source must be committed and clean")
    write_json(output / "training-plan.json", design)
    for family, process in enumerate(PILOT_LAWS):
        for cohort in (0, 1):
            seed = (TRAIN_SEED, 731, family, cohort)
            rows = simulate_events(
                np.random.default_rng(np.random.SeedSequence(seed)), process, TRAIN_N
            )
            bank = output / f"{process.kind}-{cohort}.npz"
            np.savez_compressed(
                bank,
                times=np.concatenate(rows),
                offsets=np.cumsum([0] + [len(r) for r in rows]),
            )
            try:
                fit = fit_reference(rows, process.kind)
            except Exception as error:
                write_json(
                    output / "training-error.json",
                    dict(
                        family=process.kind,
                        cohort=cohort,
                        seed_entropy=seed,
                        training_sha256=event_hash(rows),
                        error=repr(error),
                    ),
                )
                raise
            fit.update(
                protocol=PROTOCOL,
                cohort=cohort,
                seed_entropy=seed,
                family=process.kind,
                provenance=design["provenance"],
            )
            fit["bank_sha256"] = digest(bank)
            write_json(output / f"{process.kind}-{cohort}.json", fit)
            print(
                "FITTED",
                process.kind,
                cohort,
                json.dumps(fit["reference"]),
                "boundary",
                fit["boundary_parameters"],
                flush=True,
            )
    write_json(
        output / "training-complete.json",
        dict(
            status="completed",
            design=design,
            fits={
                f.name: digest(f)
                for f in sorted(output.glob("*.json"))
                if f.name != "training-complete.json"
            },
        ),
    )


def load_fit(root, kind, cohort):
    root = Path(root)
    complete = json.loads((root / "training-complete.json").read_text())
    if complete["status"] != "completed":
        raise ValueError("Training incomplete")
    for name, sha in complete["fits"].items():
        if digest(root / name) != sha:
            raise ValueError("Frozen training source changed")
    path = root / f"{kind}-{cohort}.json"
    fit = json.loads(path.read_text())
    if fit["protocol"] != PROTOCOL or fit["family"] != kind or fit["cohort"] != cohort:
        raise ValueError("Wrong frozen training reference")
    bank = root / f"{kind}-{cohort}.npz"
    if digest(bank) != fit["bank_sha256"]:
        raise ValueError("Training bank changed")
    with np.load(bank) as saved:
        rows = [
            saved["times"][a:b]
            for a, b in zip(saved["offsets"][:-1], saved["offsets"][1:])
        ]
    if len(rows) != fit["training_paths"] or event_hash(rows) != fit["training_sha256"]:
        raise ValueError("Training data hash mismatch")
    return fit, digest(path)


def replay(source, training, output, *, device="cpu"):
    source, output = Path(source), Path(output)
    old_plan = json.loads((source / "plan.json").read_text())
    plan = old_plan["source_plan"]
    fit, fit_sha = load_fit(training, CASES[plan["case"]][0].kind, plan["cohort"])
    reference = Process(**fit["reference"])
    originals = frozen_records(source / "trials.jsonl")
    mmd_source = Path(old_plan["source_directory"])
    if digest(mmd_source / "trials.jsonl") != old_plan["source_raw_sha256"]:
        raise ValueError("Archived MMD changed")
    mmd_records = frozen_records(mmd_source / "trials.jsonl")
    if len(mmd_records) != len(originals):
        raise ValueError("MMD/reference trial counts differ")
    report = json.loads((source / "report.json").read_text())
    if (
        report["status"] != "completed"
        or report["errors"]
        or len(originals) != plan["trials"]
        or any(r["status"] != "completed" for r in originals)
    ):
        raise ValueError("Complete known-reference archive required")
    if old_plan["protocol"] != "common_reference_all_event_rmse_v1":
        raise ValueError("Expected all-event archive")
    current = provenance()
    if current["tracked_dirty"]:
        raise ValueError("Replay source must be committed and clean")
    if fit["provenance"] != current:
        raise ValueError("Fit/replay source or environment differs")
    design = dict(
        protocol=PROTOCOL,
        provenance=current,
        source_directory=str(source.resolve()),
        source_raw_sha256=digest(source / "trials.jsonl"),
        source_design=old_plan,
        training_directory=str(Path(training).resolve()),
        fit_sha256=fit_sha,
        fit=fit,
        device=device,
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists() and json.loads(
        (output / "plan.json").read_text()
    ) != design:
        raise ValueError("Changed training/reference/source on resume")
    write_json(output / "plan.json", design)
    records = read_records(output / "trials.jsonl")
    if len(records) > len(originals) or any(
        r["trial"] != i or r["status"] != "completed" for i, r in enumerate(records)
    ):
        raise ValueError("Invalid replay checkpoint")
    for original in originals[len(records) :]:
        row = dict(trial=original["trial"], status="in_progress")
        try:
            mmd = mmd_records[original["trial"]]
            if original["mmd"] != mmd["methods"] or any(
                original[k] != mmd[k] for k in ("x_sha256", "y_sha256")
            ):
                raise ValueError("Known reference and MMD use different trials")
            x, y = regenerate(plan, mmd)
            if fit["training_sha256"] in (event_hash(x), event_hash(y)):
                raise ValueError("Training/evaluation bank overlap")
            scores = all_event_losses(
                x + y, reference, device=device, verify=original["trial"] == 0
            )
            n = len(x)
            row.update(
                status="completed",
                rmse=loss_test(
                    scores["path_mse"][:n],
                    scores["path_mse"][n:],
                    mmd["swap_seed"],
                    plan["permutations"],
                ),
                mmd=original["mmd"],
                known_reference=original["rmse"],
                x_sha256=original["x_sha256"],
                y_sha256=original["y_sha256"],
                swap_seed=mmd["swap_seed"],
                origins_per_path=scores["origins_per_path"].tolist(),
                events_per_path=[len(a) for a in x + y],
            )
        except Exception as error:
            row.update(status="failed", error=repr(error))
        with (output / "trials.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        records.append(row)
        if len(records) % 25 == 0:
            print(
                plan["case"],
                plan["cohort"],
                len(records),
                "/",
                len(originals),
                flush=True,
            )
    errors = [r for r in records if r["status"] != "completed"]
    result = dict(
        design=design,
        status="completed" if not errors else "completed_with_errors",
        errors=errors,
        trials=len(records),
    )
    write_json(output / "report.json", result)
    if errors:
        raise RuntimeError("Replay contains failed trials; no filtering")
    return result


def comparator(row, name):
    return row["known_reference"] if name == "known_reference" else row["mmd"][name]


def check_backend(training, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no numerical fallback")
    checks = []
    for kind in ("poisson", "exp", "power"):
        for cohort in (0, 1):
            fit, _ = load_fit(training, kind, cohort)
            # Training paths are used only to check arithmetic, not power or
            # model selection. The evaluation archives are never read here.
            with np.load(Path(training) / f"{kind}-{cohort}.npz") as bank:
                rows = [
                    bank["times"][a:b]
                    for a, b in zip(bank["offsets"][:16], bank["offsets"][1:17])
                ]
            reference = Process(**fit["reference"])
            cpu = all_event_losses(rows, reference, device="cpu", verify=True)
            gpu = all_event_losses(rows, reference, device="cuda", verify=True)
            for name in cpu:
                np.testing.assert_allclose(cpu[name], gpu[name], rtol=1e-8, atol=1e-10)
            a = loss_test(cpu["path_mse"][:8], cpu["path_mse"][8:], 2026101932, 499)
            b = loss_test(gpu["path_mse"][:8], gpu["path_mse"][8:], 2026101932, 499)
            np.testing.assert_allclose(a["null"], b["null"], rtol=1e-8, atol=1e-10)
            if a["p_value"] != b["p_value"]:
                raise ArithmeticError("Trained-reference CPU/CUDA p-values differ")
            checks.append(
                dict(
                    family=kind,
                    cohort=cohort,
                    p_value=a["p_value"],
                    boundary=fit["boundary_parameters"],
                )
            )
    write_json(
        Path(output) / "backend.json",
        dict(status="passed", device="cuda", checks=checks),
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate(root):
    root = Path(root)
    calibration, power, errors = [], [], []
    signature = None
    for case in CASES:
        pooled = []
        for cohort in (0, 1):
            try:
                directory = root / case / f"cohort-{cohort}"
                report = json.loads((directory / "report.json").read_text())
                design = report["design"]
                require(
                    design == json.loads((directory / "plan.json").read_text()),
                    "Report and plan disagree",
                )
                require(
                    design["protocol"] == PROTOCOL
                    and not design["provenance"]["tracked_dirty"],
                    "Wrong protocol or uncommitted source",
                )
                if signature is not None:
                    require(
                        signature == design["provenance"],
                        "Mixed replay environments",
                    )
                signature = design["provenance"]
                plan = design["source_design"]["source_plan"]
                for k, v in dict(
                    case=case,
                    cohort=cohort,
                    n=128,
                    permutations=499,
                    horizon=1.0,
                    trials=1000 if case.endswith("H0") else 250,
                ).items():
                    require(plan[k] == v, "Unexpected primary experimental design")
                fit, sha = load_fit(
                    design["training_directory"], CASES[case][0].kind, cohort
                )
                require(
                    fit == design["fit"] and sha == design["fit_sha256"],
                    "Frozen training fit changed",
                )
                require(
                    fit["training_paths"] == TRAIN_N
                    and fit["seed_entropy"]
                    == [
                        TRAIN_SEED,
                        731,
                        [p.kind for p in PILOT_LAWS].index(CASES[case][0].kind),
                        cohort,
                    ],
                    "Wrong training bank or seed namespace",
                )
                source = Path(design["source_directory"])
                require(
                    digest(source / "trials.jsonl") == design["source_raw_sha256"],
                    "Known-reference raw archive changed",
                )
                require(
                    json.loads((source / "plan.json").read_text())
                    == design["source_design"],
                    "Known-reference plan changed",
                )
                rows, originals = frozen_records(
                    directory / "trials.jsonl"
                ), frozen_records(source / "trials.jsonl")
                mmd_source = Path(design["source_design"]["source_directory"])
                require(
                    digest(mmd_source / "trials.jsonl")
                    == design["source_design"]["source_raw_sha256"],
                    "MMD raw archive changed",
                )
                mmd_rows = frozen_records(mmd_source / "trials.jsonl")
                require(
                    len(mmd_rows) == len(rows),
                    "MMD trial count differs",
                )
                require(
                    report["status"] == "completed"
                    and not report["errors"]
                    and len(rows) == len(originals) == plan["trials"],
                    "Incomplete or failed replay cell",
                )
                for i, (r, o) in enumerate(zip(rows, originals)):
                    require(
                        r["status"] == o["status"] == "completed"
                        and r["trial"] == o["trial"] == i,
                        "Nonmatching trial identifiers",
                    )
                    for k in (
                        "mmd",
                        "x_sha256",
                        "y_sha256",
                        "origins_per_path",
                        "events_per_path",
                    ):
                        require(
                            r[k] == o[k],
                            "Known-reference draws or trajectory identifiers changed",
                        )
                    require(
                        r["swap_seed"] == mmd_rows[i]["swap_seed"],
                        "Permutation seed changed",
                    )
                    require(
                        r["mmd"] == mmd_rows[i]["methods"],
                        "MMD statistics changed",
                    )
                    require(
                        all(r[k] == mmd_rows[i][k] for k in ("x_sha256", "y_sha256")),
                        "MMD evaluated data changed",
                    )
                    require(
                        r["known_reference"] == o["rmse"],
                        "Known-reference losses changed",
                    )
                    require(
                        r["origins_per_path"] == [n + 1 for n in r["events_per_path"]],
                        "Missing event or censored interval",
                    )
                    require(
                        len(r["origins_per_path"]) == 256,
                        "Expected 256 trajectories",
                    )
                    require(
                        r["rmse"]
                        == loss_test(
                            r["rmse"]["path_mse_x"],
                            r["rmse"]["path_mse_y"],
                            r["swap_seed"],
                            499,
                        ),
                        "Learned RMSE differs from recomputed losses and swaps",
                    )
                if case.endswith("H0"):
                    for name in ("rmse", *COMPARATORS):
                        ps = [
                            (
                                r["rmse"]["p_value"]
                                if name == "rmse"
                                else comparator(r, name)["p_value"]
                            )
                            for r in rows
                        ]
                        calibration.append(calibration_entry(case, cohort, name, ps))
                pooled.extend(rows)
            except Exception as error:
                errors.append(dict(case=case, cohort=cohort, error=repr(error)))
        if len(pooled) != (2000 if case.endswith("H0") else 500):
            continue
        if case.endswith("H0"):
            for name in ("rmse", *COMPARATORS):
                ps = [
                    (
                        r["rmse"]["p_value"]
                        if name == "rmse"
                        else comparator(r, name)["p_value"]
                    )
                    for r in pooled
                ]
                calibration.append(calibration_entry(case, "pooled", name, ps))
        else:
            for name in COMPARATORS:
                a = sum(
                    r["rmse"]["p_value"] <= 0.05
                    and comparator(r, name)["p_value"] > 0.05
                    for r in pooled
                )
                b = sum(
                    r["rmse"]["p_value"] > 0.05
                    and comparator(r, name)["p_value"] <= 0.05
                    for r in pooled
                )
                power.append(
                    dict(
                        case=case,
                        comparator=name,
                        rmse=rate_row([r["rmse"]["p_value"] for r in pooled]),
                        comparator_rate=rate_row(
                            [comparator(r, name)["p_value"] for r in pooled]
                        ),
                        rmse_only=a,
                        comparator_only=b,
                        paired_p=float(binomtest(a, a + b).pvalue) if a + b else 1.0,
                    )
                )
    for r, p in zip(power, holm_adjust([r["paired_p"] for r in power], family_size=24)):
        r["holm_p_over_24_comparisons"] = p
    complete = not errors and len(calibration) == 60 and len(power) == 24
    result = dict(
        status="complete" if complete else "incomplete",
        errors=errors,
        provenance=signature,
        calibration=calibration,
        paired_power=power,
        inference_scope="Conditional on six fixed trained references (two independent training banks per family); not unconditional across repeated training experiments",
    )
    write_json(root / "summary.json", result)
    if not complete or any(r["h0_excess_detected"] for r in calibration):
        raise RuntimeError("Campaign incomplete or H0 excess; inspect summary")
    return result


def calibration_entry(case, cohort, name, ps):
    # Reuse the verified binomial/multiplicity implementation with 60 tests.
    rows = [dict(rmse=dict(p_value=p)) for p in ps]
    result = calibration_row(case, cohort, rows, family_size=60)
    result["method"] = name
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("output", type=Path)
    p = commands.add_parser("backend")
    p.add_argument("training", type=Path)
    p.add_argument("output", type=Path)
    p = commands.add_parser("replay")
    p.add_argument("source", type=Path)
    p.add_argument("training", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p = commands.add_parser("aggregate")
    p.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train(args.output)
    elif args.command == "backend":
        check_backend(args.training, args.output)
    elif args.command == "replay":
        replay(args.source, args.training, args.output, device=args.device)
    else:
        aggregate(args.root)
