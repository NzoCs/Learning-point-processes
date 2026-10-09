"""Compare native sigkernel and pySigLib on identical seeded paths and permutations."""

import argparse
import json
from pathlib import Path
import time

import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels.signature_backend import (
    make_signature_backend,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    LinearKernel,
    RBFKernel,
)


def compare(device, repeats):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    generator = torch.Generator().manual_seed(711)
    # Generation on CPU makes the fixture identical for CPU and GPU comparisons.
    pooled = (
        torch.randn((7, 8, 3), generator=generator, dtype=torch.float64) * 0.1
    ).to(device)
    permutations = [
        torch.randperm(7, generator=generator).to(device) for _ in range(10)
    ]
    reports = []
    for static in (LinearKernel(2.5), RBFKernel(0.3, 2.5)):
        for order in (0, 1, 2):
            legacy = make_signature_backend("sigkernel", static, order, 64)
            new = make_signature_backend("pysiglib", static, order, 64)
            x, y = pooled[:4], pooled[4:]
            legacy_gram, new_gram = legacy.compute_Gram(x, y), new.compute_Gram(x, y)
            torch.testing.assert_close(new_gram, legacy_gram, rtol=1e-9, atol=1e-9)
            legacy_values, new_values = [], []
            for permutation in permutations:
                a, b = pooled[permutation[:4]], pooled[permutation[4:]]
                legacy_values.append(legacy.compute_mmd(a, b))
                new_values.append(new.compute_mmd(a, b))
            legacy_values = torch.stack(legacy_values)
            new_values = torch.stack(new_values)
            torch.testing.assert_close(new_values, legacy_values, rtol=1e-9, atol=1e-9)
            timings = {}
            for name, backend in (("sigkernel", legacy), ("pysiglib", new)):
                measurements = []
                for _ in range(repeats):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    backend.compute_Gram(x, y)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    measurements.append(time.perf_counter() - start)
                timings[name] = measurements
            reports.append(
                {
                    "static_kernel": type(static).__name__,
                    "dyadic_order": order,
                    "max_gram_absolute_error": (new_gram - legacy_gram)
                    .abs()
                    .max()
                    .item(),
                    "max_null_statistic_absolute_error": (new_values - legacy_values)
                    .abs()
                    .max()
                    .item(),
                    "legacy_null_statistics": legacy_values.cpu().tolist(),
                    "pysiglib_null_statistics": new_values.cpu().tolist(),
                    "warm_gram_seconds": timings,
                    "backend_details": new.metadata(),
                }
            )
    return {
        "device": device,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "reference_commit": "40a583155ea8d2194af0e90dddab37e2659cfcfd",
        "seed": 711,
        "rtol": 1e-9,
        "atol": 1e-9,
        "fixture": pooled.cpu().tolist(),
        "permutations": [p.cpu().tolist() for p in permutations],
        "cases": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists():
        parser.error("--output must be a new file")
    report = compare(args.device, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
