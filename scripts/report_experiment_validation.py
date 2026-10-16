"""Opt-in, matched-design validation of the report's synthetic experiments.

No model fitting or production-model changes. Independent point-process oracles,
frozen pilot normalization, fresh evaluation pairs and current paired MMD tests.
See docs/REPORT_EXPERIMENT_VALIDATION.md for the prespecified interpretation.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from new_ltpp.evaluation.statistical_testing.point_process_kernels.signature_backend import (
    PySigLibKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    LinearKernel,
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test import (
    paired_joint_statistic,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("signature_linear", "signature_rbf", "count_rbf")
MASTER_SEED = 2026101601
FAMILY_SIZE = 36  # 4 H0 cases * 3 methods * (2 cohorts + pooled).


@dataclass(frozen=True)
class Process:
    kind: str
    rate: float = 100.0  # stationary rate for Hawkes; not finite-burn rate
    branching: float = 0.5
    beta: float = 10.0
    exponent: float = 2.0


POISSON = Process("poisson")
EXP = Process("exp")
POWER = Process("power")
CASES = {
    "poisson_H0": (POISSON, POISSON, False),
    "hawkes_exp_H0": (EXP, EXP, False),
    "hawkes_power_H0": (POWER, POWER, False),
    "poisson_timing_H0": (POISSON, POISSON, True),
    "poisson_rate_110": (POISSON, Process("poisson", rate=110.0), False),
    "hawkes_alpha_8": (EXP, Process("exp", branching=0.8), False),
    "exp_vs_power_p2": (EXP, POWER, False),
    "exp_vs_power_p3": (EXP, Process("power", exponent=3.0), False),
    "hawkes_exp_timing": (EXP, EXP, True),
    "hawkes_power_timing": (POWER, POWER, True),
}
PILOT_LAWS = (POISSON, EXP, POWER)


def seed_entropy(case, cohort, trial, stream, domain=0):
    if (
        case not in CASES
        or cohort not in (0, 1)
        or trial < 0
        or stream not in (1, 2, 3)
        or domain not in (0, 1)
    ):
        raise ValueError("Invalid case/cohort/trial/stream")
    return (MASTER_SEED, 100 + domain, stream, list(CASES).index(case), cohort, trial)


def event_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(np.asarray([len(row)], dtype="<i8").tobytes())
        digest.update(np.asarray(row, dtype="<f8").tobytes())
    return digest.hexdigest()


def simulate_events(rng, process, n, horizon=1.0, burn=10.0, max_events=5_000_000):
    """Independent Poisson cluster oracle, empty history at -burn for Hawkes.

    Offspring lag density is beta*exp(-beta*t), or the Lomax density
    beta*(p-1)*(1+beta*t)^(-p). The offspring mean is the branching ratio.
    A computation-budget failure raises; it never clips events or a generation.
    """
    if (
        process.kind not in ("poisson", "exp", "power")
        or n < 1
        or not all(
            math.isfinite(v)
            for v in (
                horizon,
                burn,
                process.rate,
                process.branching,
                process.beta,
                process.exponent,
            )
        )
        or horizon <= 0
        or burn < 0
        or process.rate <= 0
        or not 0 <= process.branching < 1
        or process.beta <= 0
        or process.exponent <= 1
        or max_events < 1
    ):
        raise ValueError("Invalid independent-oracle parameters")
    if process.kind == "poisson":
        counts = rng.poisson(process.rate * horizon, n)
        if int(counts.sum()) > max_events:
            raise RuntimeError("Oracle event budget exceeded")
        return [np.sort(rng.uniform(0, horizon, int(c))) for c in counts]
    mu = process.rate * (1 - process.branching)
    counts = rng.poisson(mu * (burn + horizon), n)
    if int(counts.sum()) > max_events:
        raise RuntimeError("Oracle event budget exceeded")
    ids = np.repeat(np.arange(n), counts)
    if not len(ids):
        return [np.empty(0) for _ in range(n)]
    times = rng.uniform(-burn, horizon, len(ids))
    observed_t, observed_i, generated = [], [], len(ids)
    while len(times):
        valid = times > 0
        observed_t.append(times[valid])
        observed_i.append(ids[valid])
        offspring = rng.poisson(process.branching, len(times))
        children = int(offspring.sum())
        generated += children
        if generated > max_events:
            raise RuntimeError("Oracle event budget exceeded")
        parent_t = np.repeat(times, offspring)
        parent_i = np.repeat(ids, offspring)
        lag = (
            rng.exponential(1 / process.beta, children)
            if process.kind == "exp"
            else rng.pareto(process.exponent - 1, children) / process.beta
        )
        new_t = parent_t + lag
        keep = new_t < horizon
        times, ids = new_t[keep], parent_i[keep]
    t, index = np.concatenate(observed_t), np.concatenate(observed_i)
    order = np.lexsort((t, index))
    boundaries = np.cumsum(np.bincount(index, minlength=n))[:-1]
    return [row.copy() for row in np.split(t[order], boundaries)]


def uniformize_counts(rows, rng, horizon):
    """Condition on N(T), independently replace ONLY the event locations."""
    return [np.sort(rng.uniform(0, horizon, len(row))) for row in rows]


def prepare_paths(rows, std, grid=100, horizon=1.0):
    if not math.isfinite(std) or std <= 0 or grid < 2 or horizon <= 0:
        raise ValueError("Invalid frozen path transformation")
    times = np.linspace(0, horizon, grid)
    paths = np.empty((len(rows), grid, 2), dtype=np.float64)
    for i, row in enumerate(rows):
        if (
            not np.isfinite(row).all()
            or np.any(np.diff(row) < 0)
            or np.any(row < 0)
            or np.any(row > horizon)
        ):
            raise ValueError("Invalid observed events")
        paths[i, :, 0] = np.searchsorted(row, times, side="right") / std
    paths[:, :, 1] = times / horizon
    return torch.from_numpy(paths)


def method_grams(paths, device):
    paths = paths.to(device)
    with torch.no_grad():
        for name, static in (
            (METHODS[0], LinearKernel()),
            (METHODS[1], RBFKernel(sigma=1.0)),
        ):
            gram = PySigLibKernel(static, 0, max_batch=32).compute_Gram(paths, paths)
            if not torch.isfinite(gram).all():
                raise ValueError("Nonfinite signature Gram")
            yield name, gram
        final_count = paths[:, -1, 0]
        yield (
            METHODS[2],
            torch.exp(-(final_count[:, None] - final_count[None, :]).square()),
        )


def swap_statistics(gram, signs, *, verify=False):
    n = signs.shape[1]
    if n < 2 or gram.shape != (2 * n, 2 * n) or not torch.isfinite(gram).all():
        raise ValueError("Square finite pooled Gram and independent pairs required")
    if not ((signs == 1) | (signs == -1)).all():
        raise ValueError("Only within-pair swap signs are allowed")
    delta = gram[:n, :n] + gram[n:, n:] - gram[:n, n:] - gram[n:, :n]
    delta.fill_diagonal_(0)
    signs = signs.to(device=gram.device, dtype=torch.float64)
    values = ((signs @ delta) * signs).sum(1) / (n * (n - 1))
    if verify:
        rows = torch.arange(n, device=gram.device)
        for j in sorted({0, 1, len(signs) - 1}):
            swaps = (signs[j] < 0).long()
            reference = paired_joint_statistic(
                gram, rows + n * swaps, rows + n * (1 - swaps)
            )
            torch.testing.assert_close(values[j], reference, rtol=1e-9, atol=1e-9)
    return values.cpu()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def pilot(case, grid, horizon, burn):
    process = CASES[case][0]
    entropy = (MASTER_SEED, 0, PILOT_LAWS.index(process))
    rows = simulate_events(
        np.random.default_rng(np.random.SeedSequence(entropy)),
        process,
        512,
        horizon,
        burn,
    )
    std = float(np.std([len(r) for r in rows], ddof=1))
    if std <= 0:
        raise ValueError("Independent pilot cannot define count scale")
    return std, dict(
        entropy=list(entropy),
        event_sha256=event_hash(rows),
        count_std=std,
        paths=512,
        grid=grid,
    )


def read_records(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    # A torn final append after an interrupted job is not a completed trial.
    if raw and not raw.endswith(b"\n"):
        end = raw.rfind(b"\n") + 1
        path.with_suffix(".torn").write_bytes(raw[end:])
        raw = raw[:end]
        path.write_bytes(raw)
    rows = [json.loads(line) for line in raw.splitlines()]
    if [r["trial"] for r in rows] != list(range(len(rows))):
        raise ValueError("Checkpoint is not a contiguous, unique trial prefix")
    return rows


def summarize(case, records, planned):
    complete = len(records) == planned and all(
        r["status"] == "completed" for r in records
    )
    rows = []
    for method in METHODS:
        ps = [
            r["methods"][method]["p_value"]
            for r in records
            if r["status"] == "completed"
        ]
        rejects = sum(p <= 0.05 for p in ps)
        ci = binomtest(rejects, len(ps)).proportion_ci() if ps else None
        excess = (
            float(binomtest(rejects, len(ps), 0.05, alternative="greater").pvalue)
            if ps and case.endswith("H0")
            else None
        )
        rows.append(
            dict(
                method=method,
                planned_trials=planned,
                valid_trials=len(ps),
                rejected=rejects,
                rate=rejects / len(ps) if ps else None,
                exact_binomial_ci95=[ci.low, ci.high] if ci else None,
                h0_excess_p=excess,
                h0_excess_detected=excess is not None and excess < 0.01 / FAMILY_SIZE,
                calibration_assessable=complete and case.endswith("H0"),
            )
        )
    return rows


def run_cell(
    case,
    cohort,
    output,
    *,
    trials,
    n=128,
    permutations=499,
    grid=100,
    horizon=1.0,
    burn=10.0,
    device="cpu",
    domain=0,
):
    if (
        case not in CASES
        or cohort not in (0, 1)
        or trials < 1
        or n < 2
        or permutations < 19
        or grid < 2
        or not math.isfinite(horizon)
        or horizon <= 0
        or not math.isfinite(burn)
        or burn < 0
        or device not in ("cpu", "cuda")
        or domain not in (0, 1)
    ):
        raise ValueError("Invalid campaign cell")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    std, pilot_info = pilot(case, grid, horizon, burn)
    design = dict(
        protocol="report_matched_fixed_window_v1",
        case=case,
        cohort=cohort,
        n=n,
        trials=trials,
        permutations=permutations,
        grid=grid,
        horizon=horizon,
        burn=burn,
        master_seed=MASTER_SEED,
        evaluation_domain=domain,
        processes=[asdict(p) for p in CASES[case][:2]],
        same_counts=CASES[case][2],
        methods=list(METHODS),
        pilot=pilot_info,
        device=device,
        alpha=0.05,
        h0_family_size=FAMILY_SIZE,
        excess_threshold=0.01 / FAMILY_SIZE,
        commit=commit,
        source_lf_sha256=hashlib.sha256(
            Path(__file__).read_text().encode()
        ).hexdigest(),
        tracked_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        source_committed=subprocess.run(
            ["git", "cat-file", "-e", "HEAD:scripts/report_experiment_validation.py"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0,
        lock_sha256=hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        pysiglib=importlib.metadata.version("pysiglib"),
    )
    if (output / "plan.json").exists():
        if json.loads((output / "plan.json").read_text()) != design:
            raise ValueError(
                "Refusing to mix checkpoint with a changed protocol/source/environment"
            )
    else:
        write_json(output / "plan.json", design)
    records = read_records(output / "trials.jsonl")
    if len(records) > trials:
        raise ValueError("Checkpoint exceeds planned trials")
    started = time.monotonic()
    for trial in range(len(records), trials):
        row = dict(trial=trial, status="in_progress")
        try:
            x = simulate_events(
                np.random.default_rng(
                    np.random.SeedSequence(seed_entropy(case, cohort, trial, 1, domain))
                ),
                CASES[case][0],
                n,
                horizon,
                burn,
            )
            ry = np.random.default_rng(
                np.random.SeedSequence(seed_entropy(case, cohort, trial, 2, domain))
            )
            y = (
                uniformize_counts(x, ry, horizon)
                if CASES[case][2]
                else simulate_events(ry, CASES[case][1], n, horizon, burn)
            )
            if CASES[case][2] and [len(r) for r in x] != [len(r) for r in y]:
                raise AssertionError("Timing control changed counts")
            entropy = seed_entropy(case, cohort, trial, 3, domain)
            torch_seed = int(
                np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]
            ) % (2**63)
            generator = torch.Generator().manual_seed(torch_seed)
            signs = torch.cat(
                (
                    torch.ones(1, n),
                    1 - 2 * torch.randint(2, (permutations, n), generator=generator),
                )
            )
            row.update(
                x_sha256=event_hash(x),
                y_sha256=event_hash(y),
                swap_seed=torch_seed,
                mean_counts=[float(np.mean([len(r) for r in rows])) for rows in (x, y)],
                methods={},
            )
            z = prepare_paths(x + y, std, grid, horizon)
            for method, gram in method_grams(z, device):
                values = swap_statistics(gram, signs, verify=trial == 0)
                p = float(monte_carlo_p_value(values[0], values[1:]))
                if CASES[case][2] and method == "count_rbf" and p != 1.0:
                    raise AssertionError(
                        "Count baseline must be blind to exactly count-matched timing controls"
                    )
                row["methods"][method] = dict(
                    p_value=p, observed=float(values[0]), null=values[1:].tolist()
                )
            row["status"] = "completed"
        except Exception as error:
            row.update(status="failed", error=repr(error))
        with (output / "trials.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
        records.append(row)
        if (trial + 1) % 25 == 0 or trial + 1 == trials or row["status"] == "failed":
            write_json(
                output / "report.json",
                dict(
                    design=design,
                    status="in_progress",
                    errors=[r for r in records if r["status"] == "failed"],
                    results=summarize(case, records, trials),
                ),
            )
            print(case, cohort, trial + 1, "/", trials, row["status"], flush=True)
    results = summarize(case, records, trials)
    errors = [r for r in records if r["status"] == "failed"]
    report = dict(
        design=design,
        status="completed" if not errors else "completed_with_errors",
        errors=errors,
        results=results,
        elapsed_this_invocation=time.monotonic() - started,
    )
    write_json(output / "report.json", report)
    if errors or any(r["h0_excess_detected"] for r in results):
        raise RuntimeError("Prespecified cell failed; all trials and failures retained")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--cohort", type=int, choices=(0, 1), required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--grid", type=int, default=100)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--burn", type=float, default=10.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_cell(
        args.case,
        args.cohort,
        args.output,
        trials=args.trials,
        n=args.n,
        permutations=args.permutations,
        grid=args.grid,
        horizon=args.horizon,
        burn=args.burn,
        device=args.device,
    )


if __name__ == "__main__":
    main()
