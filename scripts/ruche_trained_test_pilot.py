"""Explicit, cost-gated Ruche pilot through the public library test APIs.

Oracle Monte Carlo controls and a trained-NHP capability probe are separate.
Real-data scoring is blocked until independent units and exposure are certified.
"""

import argparse
import hashlib
import importlib.metadata
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from new_ltpp.configs import ModelConfig
from new_ltpp.configs.window_test_config import WindowTestConfig
from new_ltpp.evaluation.benchmarks.trained_test_suite import evaluate_test_suite
from new_ltpp.evaluation.statistical_testing.references import (
    NeuralIntensityReference,
    PoissonReference,
)
from new_ltpp.models.model_factory import ModelFactory


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def model_for_rates(rates, output, device):
    import new_ltpp.models  # noqa: F401

    raw = json.loads(Path("tests/integration/config.json").read_text())["model_config"]
    raw.update(model_id="Poisson", device=device)
    raw["specs"]["rates"] = rates
    return (
        ModelFactory.create_model_by_name(
            "Poisson",
            ModelConfig.model_validate(raw),
            dict(num_event_types=len(rates), pad_token_id=len(rates), dtime_max=1.0),
            output,
        )
        .to(device)
        .eval()
    )


def configs(plan, marks):
    return [
        WindowTestConfig(horizon=1.0, num_event_types=marks, seed=plan["seed"], **m)
        for m in plan["methods"]
    ]


def evaluate(x, y, reference, settings, seconds):
    def timeout(signum, frame):
        raise TimeoutError("Prespecified full-pool cost gate exceeded")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.perf_counter()
    try:
        report = evaluate_test_suite(x, y, reference, settings)
        elapsed = time.perf_counter() - started
        if elapsed > seconds:
            raise TimeoutError("Prespecified full-pool cost gate exceeded")
        report["full_pool_seconds"] = elapsed
        return report
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def oracle(plan, case, output):
    spec = plan["oracle_cases"][case]
    rates, alternate = spec["rates"], spec["alternative_rates"]
    models = [
        model_for_rates(r, output / str(i), "cuda")
        for i, r in enumerate([rates, alternate])
    ]
    settings = configs(plan, len(rates))
    ref = PoissonReference(rates)

    def pools(n, stage, trial, hypothesis):
        drawn = []
        for stream, m in enumerate([models[0], models[hypothesis]]):
            entropy = np.random.SeedSequence(
                [plan["seed"], 908, case, stage, trial, stream]
            )
            torch.manual_seed(int(entropy.generate_state(1)[0]))
            drawn.append(m.simulate_until(n, 1.0))
        # Two-sample tests are symmetric; one-sample Stein scores the first pool.
        return drawn[::-1] if hypothesis else drawn

    warm = pools(2, 0, 0, 0)
    evaluate(*warm, ref, settings, plan["warmup_timeout_seconds"])
    for stage, n in enumerate([16, 128], start=1):
        report = evaluate(
            *pools(n, stage, 0, 0), ref, settings, plan["pool_cost_gate_seconds"]
        )
        write(output / f"probe-{n}.json", report)
    for hypothesis, label in enumerate(["h0", "h1"]):
        with (output / (label + "-trials.jsonl")).open("x") as f:
            for trial in range(plan["trials_per_hypothesis"]):
                pair = pools(128, 3, trial, hypothesis)
                for stream, data in enumerate(pair):
                    np.savez_compressed(
                        output / f"{label}-{trial:03d}-{stream}.npz",
                        times=data.time_seqs.cpu().numpy(),
                        marks=data.type_seqs.cpu().numpy(),
                        mask=data.valid_event_mask.cpu().numpy(),
                        horizon=np.array(1.0),
                    )
                report = evaluate(
                    *pair,
                    ref,
                    settings,
                    plan["pool_cost_gate_seconds"],
                )
                report.update(trial=trial, hypothesis=label, case=spec["name"])
                f.write(json.dumps(report, allow_nan=False) + "\n")
                f.flush()
    return dict(
        status="complete",
        oracle_case=spec,
        trials=2 * plan["trials_per_hypothesis"],
        n=128,
    )


def neural_probe(plan, output):
    import new_ltpp.models  # noqa: F401

    source = Path("tests/integration/baselines/NHP.json")
    raw = json.loads(Path("tests/integration/config.json").read_text())["model_config"]
    raw.update(model_id="NHP", device="cuda")
    model = ModelFactory.create_model_by_name(
        "NHP",
        ModelConfig.model_validate(raw),
        dict(num_event_types=2, pad_token_id=2, dtime_max=1.0),
        output / "model",
    )
    state = json.loads(source.read_text())["trained_state"]
    model.load_state_dict(
        {k: torch.tensor(v["values"]).reshape(v["shape"]) for k, v in state.items()},
        strict=True,
    )
    model = model.cuda().eval()
    reference = NeuralIntensityReference(model)
    generator = model_for_rates([1.0, 1.0], output / "generator", "cuda")
    torch.manual_seed(plan["seed"])
    x, y = [generator.simulate_until(16, 1.0) for _ in range(2)]
    report = evaluate(x, y, reference, configs(plan, 2), plan["pool_cost_gate_seconds"])
    report.update(
        status="capability_only_not_calibration",
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        interpretation="Existing small trained NHP fixture; input pools are Poisson, not draws from this NHP. No adequacy or power claim.",
    )
    write(output / "probe-16.json", report)
    return dict(status="capability_only_not_calibration")


def data_audit(plan, output):
    import yaml
    from datasets import load_dataset

    presets = yaml.safe_load(Path("yaml_configs/configs.yaml").read_text())[
        "data_configs"
    ]
    results = []
    for name in plan["datasets"]:
        cfg = presets[name]
        record = dict(
            dataset=name, repo=cfg["src_dir"], revision=cfg["revision"], splits={}
        )
        try:
            data = load_dataset(
                cfg["src_dir"],
                revision=cfg["revision"],
                cache_dir=str(output / "datasets"),
            )
            for split, ds in data.items():
                sample = ds[: min(32, len(ds))]
                record["splits"][split] = dict(
                    n=len(ds),
                    features=ds.features.to_dict(),
                    fingerprint=ds._fingerprint,
                    first_event_zero_count=sum(
                        bool(t) and t[0] == 0
                        for t in sample.get("time_since_start", [])
                    ),
                )
            record["status"] = "requires_exposure_and_independent_unit_certification"
            record["reason"] = (
                "Raw sequence end is not certified observation end. No fixed-window pool is exported or trained-model test run automatically."
            )
        except Exception as error:
            record.update(
                status="blocked_dataset_access",
                error_type=type(error).__name__,
                error=str(error),
            )
        results.append(record)
        write(output / (name + "-audit.json"), record)
    return dict(status="data_contract_review_required", datasets=results)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["oracle", "neural_probe", "data_audit"], required=True
    )
    parser.add_argument("--case", type=int, default=0, choices=[0, 1])
    a = parser.parse_args()
    plan = json.loads(a.plan.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    provenance = dict(
        plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        phase=a.phase,
        versions={
            n: importlib.metadata.version(n)
            for n in ["torch", "numpy", "pysiglib", "pydantic", "datasets"]
        },
    )
    write(a.output / "provenance.json", provenance)
    try:
        if a.phase != "data_audit":
            if not torch.cuda.is_available():
                raise RuntimeError("Explicit CUDA required; no CPU fallback")
            provenance["gpu"] = torch.cuda.get_device_name()
            write(
                a.output / "cuda.json",
                dict(gpu=provenance["gpu"], cuda=torch.version.cuda),
            )
        result = (
            oracle(plan, a.case, a.output)
            if a.phase == "oracle"
            else (
                neural_probe(plan, a.output)
                if a.phase == "neural_probe"
                else data_audit(plan, a.output)
            )
        )
    except Exception as error:
        write(
            a.output / "summary.json",
            dict(
                status="blocked_or_failed",
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise
    write(a.output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
