"""Frozen-kernel, nested-pool MMD versus independently trained all-event RMSE."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import beta, binomtest

from new_ltpp.evaluation.statistical_testing.point_process_kernels.signature_backend import (
    PySigLibKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from scripts.all_event_rmse import all_event_losses
from scripts.report_experiment_summary import holm_adjust
from scripts.report_experiment_validation import (
    CASES,
    ROOT,
    Process,
    event_hash,
    monte_carlo_p_value,
    pilot,
    prepare_paths,
    simulate_events,
    swap_statistics,
    uniformize_counts,
    write_json,
)
from scripts.report_learned_rmse import load_fit

PROTOCOL = "nested_pool_fixed_kernel_learned_rmse_v1"
SEED = 2026102051
SIZES = (128, 512, 1024)
METHODS = ("mmd", "rmse")
SHARDS = 5
TRAIN_COMMIT = "6ea57564a6a13efceaef02499febef7360accb70"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Require committed clean source")
    kernel = PySigLibKernel(RBFKernel(0.3), 0, max_batch=16).metadata()
    # CPU initialization/aggregation need not load a GPU driver. The installed
    # version and numerical profile remain frozen; CUDA is required by pilot/run.
    kernel.pop("cuda_plugin_loaded")
    return dict(
        commit=git("rev-parse", "HEAD"),
        source=digest(__file__),
        lock=digest(ROOT / "uv.lock"),
        torch=torch.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        python=platform.python_version(),
        kernel=kernel,
    )


def freeze(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Changed frozen contract: {path}")
    else:
        write_json(path, value)


def training_manifest(training):
    return {
        str(p): digest(p)
        for p in sorted(Path(training).iterdir())
        if p.suffix in (".npz", ".json")
    }


def initialize(root, training, selection):
    root, training, selection = Path(root), Path(training), Path(selection)
    chosen = json.loads(selection.read_text())
    if chosen["selected"]["grid100"] != "grid100_rbf0.3":
        raise ValueError("Require the already independently selected fixed RBF0.3")
    fits = {}
    for kind in ("poisson", "exp", "power"):
        for cohort in (0, 1):
            fit, sha = load_fit(training, kind, cohort)
            if (
                fit["provenance"]["commit"] != TRAIN_COMMIT
                or fit["training_paths"] != 1024
            ):
                raise ValueError("Unexpected independent training source")
            fits[f"{kind}-{cohort}"] = dict(fit=fit, sha256=sha)
    scales = {case: pilot(case, 100, 1, 10)[1] for case in CASES}
    freeze(
        root / "campaign-plan.json",
        dict(
            protocol=PROTOCOL,
            provenance=provenance(),
            seed=SEED,
            sizes=SIZES,
            permutations=499,
            grid=100,
            sigma=0.3,
            training=str(training),
            training_manifest=training_manifest(training),
            fits=fits,
            selection=str(selection),
            selection_sha256=digest(selection),
            scales=scales,
            shards=SHARDS,
            h0_trials_per_cohort=1000,
            h1_trials_per_cohort=250,
            h0_family=72,
            power_family=54,
        ),
    )


def contract(root):
    plan = json.loads((Path(root) / "campaign-plan.json").read_text())
    if (
        plan["provenance"] != provenance()
        or digest(plan["selection"]) != plan["selection_sha256"]
    ):
        raise ValueError("Source, environment or selected kernel changed")
    if training_manifest(plan["training"]) != plan["training_manifest"]:
        raise ValueError("Frozen training banks changed")
    return plan


def entropy(case, cohort, trial, stream, stage):
    if (
        case not in CASES
        or cohort not in (0, 1)
        or trial < 0
        or stream not in (1, 2, 3)
        or stage not in ("pilot", "evaluation")
    ):
        raise ValueError("Invalid independent seed coordinates")
    return (
        SEED,
        0 if stage == "pilot" else 1,
        list(CASES).index(case),
        cohort,
        trial,
        stream,
    )


def pair(case, cohort, trial, stage, n=1024):
    x = simulate_events(
        np.random.default_rng(
            np.random.SeedSequence(entropy(case, cohort, trial, 1, stage))
        ),
        CASES[case][0],
        n,
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy(case, cohort, trial, 2, stage))
    )
    y = (
        uniformize_counts(x, rng, 1)
        if CASES[case][2]
        else simulate_events(rng, CASES[case][1], n)
    )
    seed = int(
        np.random.SeedSequence(entropy(case, cohort, trial, 3, stage)).generate_state(
            1, dtype=np.uint64
        )[0]
    ) % (2**63)
    return x, y, seed


def signs(seed, n, permutations=499):
    generator = torch.Generator().manual_seed(seed)
    return torch.cat(
        (
            torch.ones(1, n),
            1 - 2 * torch.randint(2, (permutations, n), generator=generator),
        )
    ).double()


def prefix_indices(total, n, device):
    if not 2 <= n <= total:
        raise ValueError("Nested pool must fit inside full groups")
    return torch.cat(
        (torch.arange(n, device=device), torch.arange(total, total + n, device=device))
    )


def stats(values):
    return dict(
        observed=float(values[0]),
        null=values[1:].tolist(),
        p_value=float(monte_carlo_p_value(values[0], values[1:])),
    )


def evaluate(x, y, reference, std, seed, device, sizes=SIZES):
    if len(x) != len(y) or max(sizes) != len(x) or sorted(set(sizes)) != list(sizes):
        raise ValueError(
            "Ordered unique nested pool sizes and matching groups required"
        )
    started = time.monotonic()
    paths = prepare_paths(x + y, std, 100).to(device)
    with torch.no_grad():
        gram = PySigLibKernel(RBFKernel(0.3), 0, max_batch=16).compute_Gram(
            paths, paths
        )
    if not torch.isfinite(gram).all():
        raise ArithmeticError("Nonfinite Gram; no excluded trials")
    kernel_seconds = time.monotonic() - started
    scores = all_event_losses(x + y, reference, device=device)
    n = len(x)
    loss = np.asarray(scores["path_mse"])
    if loss.shape != (2 * n,) or not np.isfinite(loss).all() or np.any(loss < 0):
        raise ArithmeticError("Invalid all-event trajectory losses")
    swap = signs(seed, n)
    result = {}
    for size in sizes:
        ids = prefix_indices(n, size, gram.device)
        sub = gram.index_select(0, ids).index_select(1, ids)
        common = swap[:, :size]
        values = (
            common @ torch.from_numpy(loss[:size] - loss[n : n + size]) / size
        ).square()
        result[str(size)] = dict(
            mmd=stats(swap_statistics(sub, common)),
            rmse=stats(values),
            x_sha256=event_hash(x[:size]),
            y_sha256=event_hash(y[:size]),
        )
    return dict(
        swap_seed=seed,
        sizes=result,
        path_mse_x=loss[:n].tolist(),
        path_mse_y=loss[n:].tolist(),
        events_per_path=[len(p) for p in x + y],
        origins_per_path=scores["origins_per_path"].tolist(),
        kernel_seconds=kernel_seconds,
        total_seconds=time.monotonic() - started,
    )


def pilot_run(root):
    root = Path(root)
    plan = contract(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA pilot required; no fallback")
    checks = []
    for case in ("poisson_H0", "hawkes_exp_H0", "hawkes_power_H0"):
        for cohort in (0, 1):
            fit = plan["fits"][f"{CASES[case][0].kind}-{cohort}"]["fit"]
            reference = Process(**fit["reference"])
            x, y, seed = pair(case, cohort, 0, "pilot", n=4)
            cpu = evaluate(
                x,
                y,
                reference,
                plan["scales"][case]["count_std"],
                seed,
                "cpu",
                sizes=(2, 4),
            )
            gpu = evaluate(
                x,
                y,
                reference,
                plan["scales"][case]["count_std"],
                seed,
                "cuda",
                sizes=(2, 4),
            )
            for key in ("path_mse_x", "path_mse_y"):
                np.testing.assert_allclose(cpu[key], gpu[key], rtol=1e-8, atol=1e-10)
            for size in ("2", "4"):
                for method in METHODS:
                    a, b = cpu["sizes"][size][method], gpu["sizes"][size][method]
                    np.testing.assert_allclose(
                        [a["observed"], *a["null"]],
                        [b["observed"], *b["null"]],
                        rtol=1e-7,
                        atol=1e-8,
                    )
                    if a["p_value"] != b["p_value"]:
                        raise ArithmeticError("CPU/CUDA p-values disagree")
            checks.append(dict(case=case, cohort=cohort, cpu_cuda=True))
    timings = []
    for case in ("poisson_H0", "hawkes_exp_H0", "hawkes_power_H0"):
        reference = Process(
            **plan["fits"][f"{CASES[case][0].kind}-0"]["fit"]["reference"]
        )
        x, y, seed = pair(case, 0, 1, "pilot")
        data = evaluate(
            x, y, reference, plan["scales"][case]["count_std"], seed, "cuda"
        )
        timings.append(
            dict(
                case=case,
                kernel_seconds=data["kernel_seconds"],
                total_seconds=data["total_seconds"],
                sizes={
                    k: {m: v[m]["p_value"] for m in METHODS}
                    for k, v in data["sizes"].items()
                },
            )
        )
    freeze(
        root / "pilot.json",
        dict(
            status="completed",
            checks=checks,
            timings=timings,
            provenance=provenance(),
            device=torch.cuda.get_device_name(),
        ),
    )


def read_records(path, start=0):
    if not Path(path).exists():
        return []
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Torn output; no silent recovery")
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(r["trial"] != start + i for i, r in enumerate(rows)):
        raise ValueError("Noncontiguous trial IDs")
    return rows


def run_shard(root, index):
    root = Path(root)
    plan = contract(root)
    if not 0 <= index < 100:
        raise ValueError("Array must be 0-99")
    backend = json.loads((root / "pilot.json").read_text())
    if backend["status"] != "completed" or backend["provenance"] != provenance():
        raise ValueError("Matching successful GPU pilot required")
    case = list(CASES)[index // 10]
    cohort = index % 10 // 5
    shard = index % 5
    total = 1000 if case.endswith("H0") else 250
    start, end = shard * (total // SHARDS), (shard + 1) * (total // SHARDS)
    fit = plan["fits"][f"{CASES[case][0].kind}-{cohort}"]
    current, sha = load_fit(plan["training"], CASES[case][0].kind, cohort)
    if current != fit["fit"] or sha != fit["sha256"]:
        raise ValueError("Reference changed")
    out = root / "cells" / case / f"cohort-{cohort}" / f"shard-{shard}"
    freeze(
        out / "plan.json",
        dict(
            case=case,
            cohort=cohort,
            shard=shard,
            start=start,
            end=end,
            fit_sha256=sha,
            campaign_sha256=digest(root / "campaign-plan.json"),
        ),
    )
    rows = read_records(out / "trials.jsonl", start)
    if len(rows) > end - start:
        raise ValueError("Too many trials")
    for trial in range(start + len(rows), end):
        try:
            x, y, seed = pair(case, cohort, trial, "evaluation")
            data = evaluate(
                x,
                y,
                Process(**current["reference"]),
                plan["scales"][case]["count_std"],
                seed,
                "cuda",
            )
            data["trial"] = trial
            with (out / "trials.jsonl").open("a") as handle:
                handle.write(json.dumps(data, allow_nan=False) + "\n")
            print(case, cohort, shard, trial + 1, "/", end, flush=True)
        except Exception as error:
            write_json(out / "error.json", dict(trial=trial, error=repr(error)))
            raise
    write_json(
        out / "report.json", dict(status="completed", trials=end - start, errors=[])
    )


def proportion(k, n):
    ci = binomtest(k, n).proportion_ci()
    return dict(rejected=k, trials=n, rate=k / n, ci95=[ci.low, ci.high])


def aggregate(root):
    root = Path(root)
    plan = contract(root)
    cells = {}
    summaries = []
    comparisons = []
    excess = []
    trial_count = 0
    for case in CASES:
        for cohort in (0, 1):
            rows = []
            expected = 1000 if case.endswith("H0") else 250
            for shard in range(SHARDS):
                out = root / "cells" / case / f"cohort-{cohort}" / f"shard-{shard}"
                design = json.loads((out / "plan.json").read_text())
                report = json.loads((out / "report.json").read_text())
                start = shard * (expected // SHARDS)
                end = (shard + 1) * (expected // SHARDS)
                if (
                    design
                    != dict(
                        case=case,
                        cohort=cohort,
                        shard=shard,
                        start=start,
                        end=end,
                        fit_sha256=plan["fits"][f"{CASES[case][0].kind}-{cohort}"][
                            "sha256"
                        ],
                        campaign_sha256=digest(root / "campaign-plan.json"),
                    )
                    or report != dict(status="completed", trials=end - start, errors=[])
                    or (out / "error.json").exists()
                ):
                    raise ValueError("Incomplete or changed shard")
                part = read_records(out / "trials.jsonl", start)
                if len(part) != end - start:
                    raise ValueError("Wrong number of trials")
                rows.extend(part)
            flags = {(n, m): [] for n in SIZES for m in METHODS}
            for row in rows:
                trial = row["trial"]
                seed = int(
                    np.random.SeedSequence(
                        entropy(case, cohort, trial, 3, "evaluation")
                    ).generate_state(1, dtype=np.uint64)[0]
                ) % (2**63)
                if (
                    row["swap_seed"] != seed
                    or row["origins_per_path"]
                    != [k + 1 for k in row["events_per_path"]]
                    or len(row["origins_per_path"]) != 2048
                ):
                    raise ValueError("Changed seed or censored origins")
                loss = np.asarray([row["path_mse_x"], row["path_mse_y"]])
                swap = signs(seed, 1024)
                if (
                    loss.shape != (2, 1024)
                    or not np.isfinite(loss).all()
                    or np.any(loss < 0)
                ):
                    raise ValueError("Invalid loss vectors")
                if set(row["sizes"]) != set(map(str, SIZES)):
                    raise ValueError("Missing pool size")
                for n in SIZES:
                    expected_rmse = stats(
                        (
                            swap[:, :n]
                            @ torch.from_numpy(loss[0, :n] - loss[1, :n])
                            / n
                        ).square()
                    )
                    record = row["sizes"][str(n)]
                    if record["rmse"] != expected_rmse:
                        raise ValueError("RMSE losses/permutations do not match")
                    for m in METHODS:
                        s = record[m]
                        null = torch.tensor(s["null"], dtype=torch.float64)
                        if null.shape != (499,) or s["p_value"] != float(
                            monte_carlo_p_value(
                                torch.tensor(s["observed"], dtype=torch.float64), null
                            )
                        ):
                            raise ValueError("Invalid p-value")
                        flags[n, m].append(s["p_value"] <= 0.05)
                # Fixed endpoint samples verify deterministic trajectory replay.
                if trial % (expected // SHARDS) in (0, expected // SHARDS - 1):
                    x, y, _ = pair(case, cohort, trial, "evaluation")
                    for n in SIZES:
                        rec = row["sizes"][str(n)]
                        if rec["x_sha256"] != event_hash(x[:n]) or rec[
                            "y_sha256"
                        ] != event_hash(y[:n]):
                            raise ValueError("Nested data hash changed")
            cells[case, cohort] = {key: np.asarray(v) for key, v in flags.items()}
            trial_count += len(rows)
        for cohort in (0, 1, "pooled"):
            flags = (
                cells[case, cohort]
                if cohort != "pooled"
                else {
                    key: np.concatenate([cells[case, c][key] for c in (0, 1)])
                    for key in cells[case, 0]
                }
            )
            for n in SIZES:
                for m in METHODS:
                    v = flags[n, m]
                    entry = dict(
                        case=case,
                        cohort=cohort,
                        n=n,
                        method=m,
                        **proportion(int(v.sum()), len(v)),
                    )
                    if case.endswith("H0"):
                        p = float(
                            binomtest(
                                int(v.sum()), len(v), 0.05, alternative="greater"
                            ).pvalue
                        )
                        entry.update(excess_p=p, excess_detected=p < 0.01 / 72)
                        if entry["excess_detected"]:
                            excess.append(entry)
                        if cohort == "pooled":
                            entry["simultaneous_upper_99_over_24"] = (
                                float(
                                    beta.ppf(
                                        1 - 0.01 / 24,
                                        int(v.sum()) + 1,
                                        len(v) - int(v.sum()),
                                    )
                                )
                                if not v.all()
                                else 1.0
                            )
                    summaries.append(entry)
        if not case.endswith("H0"):
            flags = {
                key: np.concatenate([cells[case, c][key] for c in (0, 1)])
                for key in cells[case, 0]
            }
            pairs = [((n, "mmd"), (n, "rmse")) for n in SIZES] + [
                ((a, m), (b, m))
                for a, b in ((128, 512), (128, 1024), (512, 1024))
                for m in METHODS
            ]
            for a, b in pairs:
                va, vb = flags[a], flags[b]
                wins = int(np.sum(va & ~vb))
                losses = int(np.sum(~va & vb))
                p = (
                    float(binomtest(wins, wins + losses).pvalue)
                    if wins + losses
                    else 1.0
                )
                comparisons.append(
                    dict(case=case, a=a, b=b, a_only=wins, b_only=losses, p=p)
                )
    if trial_count != 11000 or len(comparisons) != 54:
        raise ValueError("Incomplete campaign")
    for row, p in zip(
        comparisons, holm_adjust([r["p"] for r in comparisons], family_size=54)
    ):
        row["holm54_p"] = p
    write_json(
        root / "summary.json",
        dict(
            status="scientific_excess" if excess else "completed",
            trials=trial_count,
            summaries=summaries,
            comparisons=comparisons,
            excess=excess,
            training_and_selection_unchanged=True,
        ),
    )
    return 2 if excess else 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=("initialize", "pilot", "run", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "initialize":
        initialize(args.root, args.training, args.selection)
    elif args.phase == "pilot":
        pilot_run(args.root)
    elif args.phase == "run":
        run_shard(args.root, args.index)
    else:
        raise SystemExit(aggregate(args.root))


if __name__ == "__main__":
    main()
