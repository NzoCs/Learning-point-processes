"""Technical CPU/CUDA check and opt-in independent local Poisson pilot.

The Poisson pilot is separate from the 11,000 archived matched MMD trials.
No model or statistic is selected from its results.
"""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from scripts.all_event_rmse import all_event_losses
from scripts.report_experiment_summary import holm_adjust
from scripts.report_experiment_validation import (
    EXP,
    POISSON,
    Process,
    event_hash,
    simulate_events,
    write_json,
)
from scripts.report_rmse_validation import landmark_losses, loss_test, rate_row

PILOT_SEED = 2026101631


def backend_check(output, device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    rng = np.random.default_rng(PILOT_SEED)
    checks = []
    for process in (POISSON, EXP, Process("power", exponent=1.5)):
        rows = simulate_events(rng, process, 16)
        start = time.monotonic()
        cpu = all_event_losses(rows, process, device="cpu", verify=True)
        target = all_event_losses(rows, process, device=device, verify=True)
        for name in cpu:
            np.testing.assert_allclose(cpu[name], target[name], rtol=1e-8, atol=1e-10)
        a = loss_test(cpu["path_mse"][:8], cpu["path_mse"][8:], 2026101632, 499)
        b = loss_test(target["path_mse"][:8], target["path_mse"][8:], 2026101632, 499)
        np.testing.assert_allclose(a["null"], b["null"], rtol=1e-8, atol=1e-10)
        if a["p_value"] != b["p_value"]:
            raise ArithmeticError("Numerical backend changed the technical p-value")
        checks.append(
            dict(
                process=process.kind,
                elapsed=time.monotonic() - start,
                p_value=a["p_value"],
                origins=int(cpu["origins_per_path"].sum()),
            )
        )
    result = dict(device=device, status="passed", checks=checks)
    write_json(Path(output) / "backend.json", result)
    return result


def poisson_pilot(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise ValueError("Use a new directory for this non-resumable local pilot")
    design = dict(
        scope="Independent local Poisson pilot, not the archived MMD comparison",
        master_seed=PILOT_SEED,
        h0_trials=1000,
        h1_trials=500,
        n=128,
        permutations=499,
        horizon=1.0,
        primary="all_event_origins_remaining_horizon",
        controls=["four_origins_remaining_horizon", "four_origins_cap_0.25"],
        numpy=np.__version__,
        torch=torch.__version__,
        sources={
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_text().encode()
            ).hexdigest()
            for name in (
                "check_all_event_rmse.py",
                "all_event_rmse.py",
                "report_rmse_validation.py",
            )
        },
    )
    write_json(output / "plan.json", design)
    reports = []
    methods = ("all_events", "four_full", "four_capped")
    errors = []
    with gzip.open(output / "trials.jsonl.gz", "wt", encoding="utf-8") as handle:
        for case, trials, alternative in (
            ("H0", 1000, POISSON),
            ("rate_110", 500, Process("poisson", rate=110)),
        ):
            records = []
            for trial in range(trials):
                row = dict(case=case, trial=trial, status="in_progress")
                try:
                    rng_x = np.random.default_rng(
                        np.random.SeedSequence(
                            (PILOT_SEED, 0 if case == "H0" else 1, trial, 1)
                        )
                    )
                    rng_y = np.random.default_rng(
                        np.random.SeedSequence(
                            (PILOT_SEED, 0 if case == "H0" else 1, trial, 2)
                        )
                    )
                    x = simulate_events(rng_x, POISSON, 128)
                    y = simulate_events(rng_y, alternative, 128)
                    seed = int(
                        np.random.SeedSequence(
                            (PILOT_SEED, 0 if case == "H0" else 1, trial, 3)
                        ).generate_state(1, dtype=np.uint64)[0]
                    ) % (2**63)
                    scores = all_event_losses(x + y, POISSON)
                    capped = landmark_losses(x + y, POISSON)
                    row.update(
                        status="completed",
                        x_sha256=event_hash(x),
                        y_sha256=event_hash(y),
                        swap_seed=seed,
                        methods={
                            name: loss_test(values[:128], values[128:], seed, 499)
                            for name, values in zip(
                                methods,
                                (
                                    scores["path_mse"],
                                    scores["landmark_full_mse"],
                                    capped,
                                ),
                            )
                        },
                    )
                except Exception as error:
                    row.update(status="failed", error=repr(error))
                    errors.append(row)
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                records.append(row)
                if (trial + 1) % 100 == 0:
                    print(case, trial + 1, "/", trials, flush=True)
            rates = (
                {
                    name: rate_row(
                        [
                            r["methods"][name]["p_value"]
                            for r in records
                            if r["status"] == "completed"
                        ]
                    )
                    for name in methods
                }
                if not errors
                else {}
            )
            comparisons = []
            if case != "H0" and not errors:
                for name in methods[1:]:
                    a = sum(
                        r["methods"]["all_events"]["p_value"] <= 0.05
                        and r["methods"][name]["p_value"] > 0.05
                        for r in records
                    )
                    b = sum(
                        r["methods"]["all_events"]["p_value"] > 0.05
                        and r["methods"][name]["p_value"] <= 0.05
                        for r in records
                    )
                    comparisons.append(
                        dict(
                            control=name,
                            all_only=a,
                            control_only=b,
                            paired_p=(
                                float(binomtest(a, a + b).pvalue) if a + b else 1.0
                            ),
                        )
                    )
                for r, p in zip(
                    comparisons,
                    holm_adjust([r["paired_p"] for r in comparisons], family_size=2),
                ):
                    r["holm_p_over_2"] = p
            reports.append(dict(case=case, rates=rates, paired_comparisons=comparisons))
            write_json(
                output / "report.json",
                dict(
                    design=design, status="in_progress", errors=errors, results=reports
                ),
            )
    result = dict(
        design=design,
        status="complete" if not errors else "completed_with_errors",
        errors=errors,
        results=reports,
    )
    write_json(output / "report.json", result)
    if errors:
        raise RuntimeError("Pilot contains failed trials; no filtering")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poisson-pilot", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.poisson_pilot:
        if args.device != "cpu":
            raise ValueError("Local Poisson pilot uses the analytic CPU calculation")
        poisson_pilot(args.output)
    else:
        backend_check(args.output, args.device)
