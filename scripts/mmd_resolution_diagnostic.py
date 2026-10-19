"""Opt-in, independent selection and matched replay of MMD representations.

No production defaults or input archives are changed. See REPORT_MMD_RESOLUTION.md.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    LinearKernel,
    RBFKernel,
)
from scripts.report_experiment_summary import holm_adjust
from scripts.report_experiment_validation import (
    CASES,
    ROOT,
    PySigLibKernel,
    event_hash,
    monte_carlo_p_value,
    pilot,
    prepare_paths,
    simulate_events,
    swap_statistics,
    uniformize_counts,
    write_json,
)
from scripts.report_rmse_validation import regenerate

SEED = 2026102037
REPS = ("grid100", "grid400", "events")
KERNELS = ("linear", "rbf0.3", "rbf1", "rbf3")
CANDIDATES = tuple(f"{r}_{k}" for r in REPS for k in KERNELS)
BASELINES = ("grid100_linear", "grid100_rbf1")
H0 = tuple(c for c in CASES if c.endswith("H0"))
H1 = tuple(c for c in CASES if not c.endswith("H0"))
PROTOCOL = "independent_selection_matched_mmd_resolution_v1"
SOURCE_COMMIT = "9398381cb6d2b71b618b37d78299c7501d121f80"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_rows(rows):
    if not rows:
        raise ValueError("Empty path bank")
    for row in rows:
        if (
            np.asarray(row).ndim != 1
            or not np.isfinite(row).all()
            or np.any(np.diff(row) < 0)
            or np.any(row <= 0)
            or np.any(row >= 1)
        ):
            raise ValueError("Events must be ordered and strictly inside (0,1)")


def event_paths(rows, std):
    """Time then counting jump at each event; pad by repeating (N/std,1).

    This preserves all jump vertices, not an exact solution of the signature PDE.
    The same finite-difference solver/order as the baseline is retained.
    """
    checked_rows(rows)
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Positive finite common count scale required")
    length = 2 * max(map(len, rows)) + 2
    paths = np.empty((len(rows), length, 2), dtype=np.float64)
    for i, row in enumerate(rows):
        paths[i, 0] = (0, 0)
        for j, t in enumerate(row):
            paths[i, 2 * j + 1] = (j / std, t)
            paths[i, 2 * j + 2] = ((j + 1) / std, t)
        paths[i, 2 * len(row) + 1 :] = (len(row) / std, 1)
    return torch.from_numpy(paths)


def signs_for(seed, n=128, permutations=499):
    generator = torch.Generator().manual_seed(seed)
    return torch.cat(
        (
            torch.ones(1, n),
            1 - 2 * torch.randint(2, (permutations, n), generator=generator),
        )
    )


def kernel_gram(paths, name, order=0):
    static = LinearKernel() if name == "linear" else RBFKernel(float(name[3:]))
    with torch.no_grad():
        gram = PySigLibKernel(static, order, max_batch=16).compute_Gram(paths, paths)
    if not torch.isfinite(gram).all():
        raise ArithmeticError("Nonfinite Gram; no trial may be dropped")
    return gram


def evaluate(x, y, std, seed, methods, device):
    checked_rows(x)
    checked_rows(y)
    if len(x) != len(y):
        raise ValueError("Matched groups of equal size required")
    signs = signs_for(seed, len(x))
    result = {}
    for rep in REPS:
        names = [m for m in methods if m.startswith(rep + "_")]
        if not names:
            continue
        rows = x + y
        paths = (
            event_paths(rows, std)
            if rep == "events"
            else prepare_paths(rows, std, int(rep[4:]))
        ).to(device)
        for name in names:
            started = time.monotonic()
            gram = kernel_gram(paths, name.split("_", 1)[1])
            values = swap_statistics(gram, signs)
            result[name] = dict(
                p_value=float(monte_carlo_p_value(values[0], values[1:])),
                observed=float(values[0]),
                null=values[1:].tolist(),
                vertices=paths.shape[1],
                seconds=time.monotonic() - started,
            )
    return result


def independent_pair(case, trial, stage, n=128):
    if stage not in ("selection", "calibration") or trial < 0:
        raise ValueError("Separate prespecified stages required")
    namespace = 1 if stage == "selection" else 2
    entropy = (SEED, namespace, list(CASES).index(case), trial)
    x = simulate_events(
        np.random.default_rng(np.random.SeedSequence((*entropy, 1))), CASES[case][0], n
    )
    rng = np.random.default_rng(np.random.SeedSequence((*entropy, 2)))
    y = (
        uniformize_counts(x, rng, 1)
        if CASES[case][2]
        else simulate_events(rng, CASES[case][1], n)
    )
    seed = int(
        np.random.SeedSequence((*entropy, 3)).generate_state(1, dtype=np.uint64)[0]
    ) % (2**63)
    return x, y, seed


def source_manifest(source):
    source = Path(source)
    files = sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and p.name in ("plan.json", "report.json", "trials.jsonl", "summary.json")
    )
    return {str(p.relative_to(source)): sha(p) for p in files}


def provenance():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise ValueError("Require committed clean numerical source")
    return dict(
        commit=commit,
        torch=torch.__version__,
        numpy=np.__version__,
        lock_sha256=sha(ROOT / "uv.lock"),
        script_sha256=sha(__file__),
    )


def freeze(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing changed contract: {path}")
    else:
        write_json(path, value)


def initialize(root, source):
    manifest = source_manifest(source)
    if len(manifest) != 61:
        raise ValueError("Require all 20 immutable source cells and summary")
    root = Path(root)
    for case in CASES:
        for cohort in (0, 1):
            p = Path(source) / case / f"cohort-{cohort}"
            plan = json.loads((p / "plan.json").read_text())
            report = json.loads((p / "report.json").read_text())
            expected = 1000 if case in H0 else 250
            if (
                plan["commit"] != SOURCE_COMMIT
                or plan["n"] != 128
                or plan["grid"] != 100
                or plan["permutations"] != 499
                or plan["trials"] != expected
                or plan["evaluation_domain"] != 0
                or report["status"] != "completed"
                or report.get("errors")
            ):
                raise ValueError("Unexpected or incomplete MMD source")
    freeze(root / "source-manifest.json", manifest)
    freeze(
        root / "campaign-plan.json",
        dict(
            protocol=PROTOCOL,
            provenance=provenance(),
            source=str(source),
            seed=SEED,
            candidates=CANDIDATES,
            selection_trials_per_case=32,
            fresh_h0_trials_per_case=1000,
            archived_trials=11000,
            n=128,
            permutations=499,
            baseline_methods=BASELINES,
            selection_rule="per representation: maximum mean rejection rate over six H1 cases; ties follow fixed candidate order; no calibration filtering",
            h0_family_size=48,
            paired_family_size=72,
            input_manifest_sha256=sha(root / "source-manifest.json"),
        ),
    )


def append_trial(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def records(path):
    if not Path(path).exists():
        return []
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(
            "Torn output requires explicit recovery; never silently exclude"
        )
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(r["trial"] != i for i, r in enumerate(rows)):
        raise ValueError("Noncontiguous trial archive")
    for row in rows:
        for method in row["methods"].values():
            p = method["p_value"]
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Invalid p-value")
            if "null" in method and p != float(
                monte_carlo_p_value(
                    torch.tensor(method["observed"], dtype=torch.float64),
                    torch.tensor(method["null"], dtype=torch.float64),
                )
            ):
                raise ValueError("P-value does not match archived null statistics")
    return rows


def run_independent(root, case, stage, device):
    root = Path(root)
    selection = (
        json.loads((root / "selection.json").read_text())
        if stage == "calibration"
        else None
    )
    methods = (
        list(dict.fromkeys((*BASELINES, *selection["selected"].values())))
        if selection
        else list(CANDIDATES)
    )
    trials = 1000 if selection else 32
    if stage == "calibration" and case not in H0:
        raise ValueError("Independent confirmation is H0 only")
    std, info = pilot(case, 100, 1, 10)
    out = root / stage / case
    freeze(
        out / "plan.json",
        dict(
            case=case,
            stage=stage,
            trials=trials,
            methods=methods,
            pilot=info,
            provenance=provenance(),
            selection_sha256=sha(root / "selection.json") if selection else None,
        ),
    )
    previous = records(out / "trials.jsonl")
    if len(previous) > trials:
        raise ValueError("Too many trials")
    for trial in range(len(previous), trials):
        x, y, seed = independent_pair(case, trial, stage)
        row = dict(
            trial=trial,
            x_sha256=event_hash(x),
            y_sha256=event_hash(y),
            swap_seed=seed,
            methods=evaluate(x, y, std, seed, methods, device),
        )
        append_trial(out / "trials.jsonl", row)
        print(stage, case, trial + 1, "/", trials, flush=True)
    write_json(out / "report.json", dict(status="completed", trials=trials, errors=[]))


def select(root):
    root = Path(root)
    rates = {name: [] for name in CANDIDATES}
    for case in CASES:
        rows = records(root / "selection" / case / "trials.jsonl")
        if len(rows) != 32:
            raise ValueError("All independent selection cells required")
        for name in CANDIDATES:
            if case in H1:
                rates[name].append(
                    sum(r["methods"][name]["p_value"] <= 0.05 for r in rows) / 32
                )
    scores = {name: float(np.mean(values)) for name, values in rates.items()}
    chosen = {
        rep: max(
            (m for m in CANDIDATES if m.startswith(rep + "_")), key=lambda m: scores[m]
        )
        for rep in REPS
    }
    freeze(
        root / "selection.json",
        dict(
            protocol=PROTOCOL,
            selected=chosen,
            scores=scores,
            selection_manifest=source_trial_manifest(root / "selection"),
            selection_evaluation_namespace=1,
        ),
    )


def source_trial_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.json*"))}


def validate_frozen(root):
    root = Path(root)
    campaign = json.loads((root / "campaign-plan.json").read_text())
    if campaign["provenance"] != provenance():
        raise ValueError("Frozen diagnostic source/environment changed")
    if sha(root / "source-manifest.json") != campaign["input_manifest_sha256"]:
        raise ValueError("Input manifest changed")


def run_pilot(root, device):
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("Explicit CUDA pilot required before the diagnostic")
    root = Path(root)
    x, y, seed = independent_pair("hawkes_exp_H0", 0, "selection", n=4)
    std, _ = pilot("hawkes_exp_H0", 100, 1, 10)
    cpu = evaluate(x, y, std, seed, CANDIDATES, "cpu")
    cuda = evaluate(x, y, std, seed, CANDIDATES, "cuda")
    for name in CANDIDATES:
        np.testing.assert_allclose(
            [cpu[name]["observed"], *cpu[name]["null"]],
            [cuda[name]["observed"], *cuda[name]["null"]],
            rtol=1e-7,
            atol=1e-8,
        )
        if cpu[name]["p_value"] != cuda[name]["p_value"]:
            raise ArithmeticError("CPU/CUDA p-value mismatch")
    refinement = {}
    for rep in ("grid100", "events"):
        paths = (
            event_paths(x + y, std)
            if rep == "events"
            else prepare_paths(x + y, std, 100)
        ).to(device)
        for kernel in ("linear", "rbf1"):
            grams = [
                kernel_gram(paths, kernel, order=order).cpu() for order in (0, 1, 2)
            ]
            refinement[f"{rep}_{kernel}"] = [
                float(torch.linalg.norm(a - b) / torch.linalg.norm(b))
                for a, b in zip(grams[:-1], grams[1:])
            ]
    campaign = json.loads((root / "campaign-plan.json").read_text())
    source = Path(campaign["source"]) / "hawkes_exp_H0" / "cohort-0"
    plan = json.loads((source / "plan.json").read_text())
    with (source / "trials.jsonl").open() as handle:
        row = json.loads(handle.readline())
    a, b = regenerate(plan, row)
    measured = evaluate(
        a, b, plan["pilot"]["count_std"], row["swap_seed"], CANDIDATES, device
    )
    for name, old in zip(BASELINES, ("signature_linear", "signature_rbf")):
        if measured[name]["p_value"] != row["methods"][old]["p_value"]:
            raise ArithmeticError("Historical baseline not reproduced")
        np.testing.assert_allclose(
            [measured[name]["observed"], *measured[name]["null"]],
            [row["methods"][old]["observed"], *row["methods"][old]["null"]],
            rtol=1e-8,
            atol=1e-8,
        )
    write_json(
        root / "pilot.json",
        dict(
            status="completed",
            cpu_cuda_p_values_equal=True,
            cpu=cpu,
            cuda=cuda,
            solver_refinement_relative_gram_differences=refinement,
            representative_n128=measured,
            baseline_reproduced=True,
            device=torch.cuda.get_device_name(),
            provenance=provenance(),
        ),
    )


def replay(root, case, cohort, device):
    root = Path(root)
    campaign = json.loads((root / "campaign-plan.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    methods = list(
        dict.fromkeys(
            [f"{rep}_{k}" for rep in REPS for k in ("linear", "rbf1")]
            + list(selection["selected"].values())
        )
    )
    source = Path(campaign["source"]) / case / f"cohort-{cohort}"
    plan = json.loads((source / "plan.json").read_text())
    old = records(source / "trials.jsonl")
    if len(old) != plan["trials"]:
        raise ValueError("Incomplete input")
    out = root / "replay" / case / f"cohort-{cohort}"
    freeze(
        out / "plan.json",
        dict(
            source_plan=plan,
            methods=methods,
            selection_sha256=sha(root / "selection.json"),
            provenance=provenance(),
        ),
    )
    previous = records(out / "trials.jsonl")
    if len(previous) > len(old):
        raise ValueError("Too many replay trials")
    for row in old[len(previous) :]:
        x, y = regenerate(plan, row)
        result = evaluate(
            x, y, plan["pilot"]["count_std"], row["swap_seed"], methods, device
        )
        for name, source_name in zip(BASELINES, ("signature_linear", "signature_rbf")):
            a, b = result[name], row["methods"][source_name]
            np.testing.assert_allclose(
                [a["observed"], *a["null"]],
                [b["observed"], *b["null"]],
                rtol=1e-8,
                atol=1e-8,
            )
            if a["p_value"] != b["p_value"]:
                raise ArithmeticError("Baseline p-value changed")
        append_trial(
            out / "trials.jsonl",
            dict(
                trial=row["trial"],
                x_sha256=row["x_sha256"],
                y_sha256=row["y_sha256"],
                swap_seed=row["swap_seed"],
                methods=result,
            ),
        )
        print("replay", case, cohort, row["trial"] + 1, "/", len(old), flush=True)
    write_json(
        out / "report.json", dict(status="completed", trials=len(old), errors=[])
    )


def describe(ps, h0=False):
    k, n = sum(p <= 0.05 for p in ps), len(ps)
    interval = binomtest(k, n).proportion_ci()
    return dict(
        rejected=k,
        trials=n,
        rate=k / n,
        ci95=[interval.low, interval.high],
        h0_excess_p=(
            float(binomtest(k, n, 0.05, alternative="greater").pvalue) if h0 else None
        ),
    )


def aggregate(root):
    root = Path(root)
    campaign = json.loads((root / "campaign-plan.json").read_text())
    if source_manifest(campaign["source"]) != json.loads(
        (root / "source-manifest.json").read_text()
    ):
        raise ValueError("Source archive changed")
    selection = json.loads((root / "selection.json").read_text())
    if source_trial_manifest(root / "selection") != selection["selection_manifest"]:
        raise ValueError("Selection data changed")
    results, comparisons, calibrations = {}, [], []
    for case in CASES:
        pooled = []
        for cohort in (0, 1):
            cell = root / "replay" / case / f"cohort-{cohort}"
            rows = records(cell / "trials.jsonl")
            if len(rows) != (1000 if case in H0 else 250):
                raise ValueError("Incomplete replay")
            plan = json.loads((cell / "plan.json").read_text())
            if (
                plan["selection_sha256"] != sha(root / "selection.json")
                or plan["provenance"] != provenance()
            ):
                raise ValueError("Replay selection/source changed")
            old = records(
                Path(campaign["source"]) / case / f"cohort-{cohort}" / "trials.jsonl"
            )
            for a, b in zip(rows, old, strict=True):
                if any(
                    a[k] != b[k] for k in ("trial", "x_sha256", "y_sha256", "swap_seed")
                ):
                    raise ValueError("Unmatched evaluation trajectories/permutations")
                if set(a["methods"]) != set(plan["methods"]):
                    raise ValueError("Missing diagnostic methods")
                for name, source_name in zip(
                    BASELINES, ("signature_linear", "signature_rbf")
                ):
                    if (
                        a["methods"][name]["p_value"]
                        != b["methods"][source_name]["p_value"]
                    ):
                        raise ValueError("Historical baseline changed")
            pooled.extend(rows)
            results[f"{case}/cohort-{cohort}"] = {
                m: describe([r["methods"][m]["p_value"] for r in rows], case in H0)
                for m in rows[0]["methods"]
            }
        results[case] = {
            m: describe([r["methods"][m]["p_value"] for r in pooled], case in H0)
            for m in pooled[0]["methods"]
        }
        if case in H1:
            pairs = [
                (f"{rep}_{k}", f"grid100_{k}")
                for rep in ("grid400", "events")
                for k in ("linear", "rbf1")
            ]
            pairs += [
                (selection["selected"][rep], f"{rep}_{k}")
                for rep in REPS
                for k in ("linear", "rbf1")
            ]
            pairs += [
                (selection["selected"][rep], selection["selected"]["grid100"])
                for rep in ("grid400", "events")
            ]
            for a, b in pairs:
                wins = sum(
                    r["methods"][a]["p_value"] <= 0.05 < r["methods"][b]["p_value"]
                    for r in pooled
                )
                losses = sum(
                    r["methods"][b]["p_value"] <= 0.05 < r["methods"][a]["p_value"]
                    for r in pooled
                )
                comparisons.append(
                    dict(
                        case=case,
                        a=a,
                        b=b,
                        a_only=wins,
                        b_only=losses,
                        p=(
                            float(binomtest(wins, wins + losses).pvalue)
                            if wins + losses
                            else 1.0
                        ),
                    )
                )
    for case in H0:
        cell = root / "calibration" / case
        rows = records(cell / "trials.jsonl")
        if len(rows) != 1000:
            raise ValueError("Incomplete independent calibration")
        plan = json.loads((cell / "plan.json").read_text())
        if (
            plan["selection_sha256"] != sha(root / "selection.json")
            or plan["provenance"] != provenance()
        ):
            raise ValueError("Calibration selection/source changed")
        for name in rows[0]["methods"]:
            calibrations.append(
                dict(
                    case=case,
                    method=name,
                    **describe([r["methods"][name]["p_value"] for r in rows], True),
                )
            )
    # Fixed maximum families include duplicate methods from equal selections.
    adjusted = holm_adjust([c["p"] for c in comparisons], family_size=72)
    for row, value in zip(comparisons, adjusted):
        row["holm72_p"] = value
    excess = [c for c in calibrations if c["h0_excess_p"] < 0.01 / 48]
    write_json(
        root / "summary.json",
        dict(
            status="scientific_excess" if excess else "completed",
            archived_trials=11000,
            fresh_h0_trials=4000,
            selection=selection,
            results=results,
            independent_calibration=calibrations,
            mcnemar=comparisons,
            scientific_excess=excess,
            input_archives_unchanged=True,
            inference="Exploratory matched archive replay; independent H0 confirmation; no universal superiority or model validation",
        ),
    )
    return 2 if excess else 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "initialize",
            "pilot",
            "selection",
            "select",
            "calibration",
            "replay",
            "aggregate",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--cohort", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.phase == "initialize":
        initialize(args.root, args.source)
        return
    validate_frozen(args.root)
    if args.phase == "pilot":
        run_pilot(args.root, args.device)
        return
    backend = json.loads((args.root / "pilot.json").read_text())
    if backend["status"] != "completed" or backend["provenance"] != provenance():
        raise ValueError("Successful matching CUDA pilot required")
    if args.phase in ("selection", "calibration"):
        run_independent(args.root, args.case, args.phase, args.device)
    elif args.phase == "select":
        select(args.root)
    elif args.phase == "replay":
        replay(args.root, args.case, args.cohort, args.device)
    else:
        raise SystemExit(aggregate(args.root))


if __name__ == "__main__":
    main()
