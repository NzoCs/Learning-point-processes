"""Technical CPU/CUDA agreement and representative timing; no calibration claim."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.report_experiment_validation import (
    POISSON,
    method_grams,
    pilot,
    prepare_paths,
    run_cell,
    simulate_events,
    swap_statistics,
    write_json,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)


def check_backend(output, device="cuda"):
    output = Path(output)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    std, _ = pilot("poisson_H0", 100, 1.0, 10.0)
    rng = np.random.default_rng(2026101698)
    paths = prepare_paths(simulate_events(rng, POISSON, 16), std, 100)
    signs = torch.cat(
        (
            torch.ones(1, 8),
            1
            - 2
            * torch.randint(
                2, (99, 8), generator=torch.Generator().manual_seed(2026101699)
            ),
        )
    )
    cpu = dict(method_grams(paths, "cpu"))
    checks = []
    for method, gram in method_grams(paths, device):
        torch.testing.assert_close(gram.cpu(), cpu[method], rtol=1e-8, atol=1e-8)
        a = swap_statistics(cpu[method], signs, verify=True)
        b = swap_statistics(gram, signs, verify=True)
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-7)
        checks.append(
            dict(
                method=method,
                max_abs_gram_difference=float((gram.cpu() - cpu[method]).abs().max()),
                max_relative_gram_difference=float(
                    (
                        (gram.cpu() - cpu[method]).abs()
                        / cpu[method].abs().clamp_min(1)
                    ).max()
                ),
                max_abs_statistic_difference=float((a - b).abs().max()),
                cpu_p=float(monte_carlo_p_value(a[0], a[1:])),
                target_p=float(monte_carlo_p_value(b[0], b[1:])),
            )
        )
    report = dict(
        device=device,
        checks=checks,
        scope="Technical check only, not a size/power estimate",
    )
    write_json(output / "backend.json", report)
    times = []
    for case in ("poisson_H0", "hawkes_exp_timing", "exp_vs_power_p3"):
        r = run_cell(case, 0, output / case, trials=2, device=device, domain=1)
        times.append(
            dict(
                case=case, trials=2, seconds_per_trial=r["elapsed_this_invocation"] / 2
            )
        )
    report["timing"] = times
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    write_json(output / "backend.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(check_backend(args.output, args.device), indent=2), flush=True)
