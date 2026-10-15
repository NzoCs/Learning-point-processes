"""Matched fixed-horizon CPU library comparisons, and synchronized CUDA when available."""

import argparse
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import ks_2samp

from scripts.integration_suite import write_json


def reference(name, count, horizon, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        if name == "Hawkes":
            from HawkesPyLib.simulation import ExpHawkesProcessSimulation

            rows.append(
                ExpHawkesProcessSimulation(mu=1.2, eta=0.3, theta=1 / 1.4).simulate(
                    horizon, seed=seed + i
                )
            )
        elif name == "Poisson":
            rows.append(np.sort(rng.uniform(0, horizon, rng.poisson(1.2 * horizon))))
        else:
            # SciPy numerical inversion is independent of the Torch closed-form inverse.
            t, penalty = 0.0, 0.0
            events = []
            while True:
                exponential = rng.exponential()
                rate = math.exp(0.2 * t - penalty)
                bound = 1.0
                while rate * math.expm1(0.2 * bound) / 0.2 < exponential:
                    bound *= 2
                delta = brentq(
                    lambda d: rate * math.expm1(0.2 * d) / 0.2 - exponential, 0, bound
                )
                t += delta
                if t >= horizon:
                    break
                events.append(t)
                penalty += 0.5
            rows.append(np.array(events))
    return rows


def measure(call, device, repeats=5):
    call()  # allocation/JIT warmup, excluded
    durations = []
    result = None
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = call()
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return result, dict(
        median_seconds=float(np.median(durations)),
        min_seconds=min(durations),
        max_seconds=max(durations),
        repeats=repeats,
        warmup_excluded=True,
    )


def run(output, device="auto", samples=1024, horizon=20.0, repeats=5):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA GPU is available")
    devices = (
        ["cpu", "cuda"]
        if device == "auto" and torch.cuda.is_available()
        else ["cpu" if device == "auto" else device]
    )
    report = dict(
        status="in_progress",
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        gpu_status="measured" if "cuda" in devices else "not_measured_no_cuda",
        libraries={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "HawkesPyLib", "numba")
        },
        task="univariate processes from empty history on the same fixed horizon",
        horizon=horizon,
        sequences_for_distribution=samples,
        results={},
    )
    for name in ("Hawkes", "SelfCorrecting", "Poisson"):
        params = (
            dict(mu=[1.2], alpha=[[0.3]], beta=[[1.4]])
            if name == "Hawkes"
            else (
                dict(mu=[0.2], alpha=[[0.5]])
                if name == "SelfCorrecting"
                else dict(rates=[1.2])
            )
        )
        # Factory fixture is marked K=2; construct this explicitly comparable univariate case.
        import new_ltpp.models  # noqa: F401
        from new_ltpp.configs import ModelConfig
        from new_ltpp.models.model_factory import ModelFactory

        raw = json.loads(Path("tests/integration/config.json").read_text())[
            "model_config"
        ]
        raw["model_id"] = name

        def make(dev):
            return (
                ModelFactory.create_model_by_name(
                    name,
                    ModelConfig.model_validate(raw),
                    dict(
                        num_event_types=1,
                        pad_token_id=1,
                        end_time_max=horizon,
                        dtime_max=horizon,
                    ),
                    output / name,
                    **params,
                )
                .to(dev)
                .eval()
            )

        cpu_model = make("cpu")
        torch.manual_seed(20261021)
        trajectories = cpu_model.simulate_until(samples, horizon)
        known = reference(name, samples, horizon, 20261022)
        counts = trajectories.valid_event_mask.sum(1).numpy()
        ref_counts = np.array([len(row) for row in known])
        distance = float(ks_2samp(counts, ref_counts).statistic)
        bound = math.sqrt(2 * math.log(4 / 1e-4) / samples)
        rows = [
            trajectories.time_seqs[i][trajectories.valid_event_mask[i]]
            .numpy()
            .astype(float)
            for i in range(samples)
        ]
        # Counts and first-event CDF at fixed quantiles use equal observation windows, no count filtering.
        first = np.array([row[0] if len(row) else np.inf for row in rows])
        other = np.array([row[0] if len(row) else np.inf for row in known])
        first_difference = max(
            abs(float((first <= t).mean()) - float((other <= t).mean()))
            for t in (0.1, 0.5, 1.0, 2.0, 5.0)
        )
        result = dict(
            reference=(
                "HawkesPyLib 0.3.0"
                if name == "Hawkes"
                else (
                    "SciPy brentq + NumPy exponential (numerical oracle, not a turnkey SelfCorrecting library)"
                    if name == "SelfCorrecting"
                    else "NumPy poisson + uniform sorted"
                )
            ),
            parameters=params,
            consistency=dict(
                passed=distance <= bound and first_difference <= bound,
                count_ks=distance,
                two_sample_dkw_bound=bound,
                mean_events_torch=float(counts.mean()),
                mean_events_reference=float(ref_counts.mean()),
                first_event_cdf_max_difference=first_difference,
            ),
            timings=[],
        )
        for batch_size in (1, 128):
            _, timing = measure(
                lambda: reference(name, batch_size, horizon, 20261023), "cpu", repeats
            )
            result["timings"].append(
                dict(
                    implementation="reference",
                    device="cpu",
                    batch_size=batch_size,
                    **timing,
                )
            )
            for dev in devices:
                model = cpu_model if dev == "cpu" else make(dev)
                generated, timing = measure(
                    lambda: model.simulate_until(batch_size, horizon), dev, repeats
                )
                assert generated.time_seqs.device.type == dev
                assert torch.isfinite(generated.time_seqs).all()
                result["timings"].append(
                    dict(
                        implementation="new_ltpp",
                        device=dev,
                        batch_size=batch_size,
                        events=int(generated.valid_event_mask.sum()),
                        host_transfer_excluded=True,
                        **timing,
                    )
                )
        report["results"][name] = result
        write_json(output / "report.json", report)
        print(name, result, flush=True)
    report["status"] = "completed"
    assert all(
        value["consistency"]["passed"] for value in report["results"].values()
    ), "Independent library distribution mismatch"
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/parametric-benchmark")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--horizon", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=5)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
