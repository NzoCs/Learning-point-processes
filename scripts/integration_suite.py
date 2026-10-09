"""Opt-in, offline end-to-end regression suite for every public TPP model."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "integration"
MODELS = (
    "ANHN",
    "ANHP",
    "FullyNN",
    "Hawkes",
    "IntensityFree",
    "NHP",
    "ODETPP",
    "RMTPP",
    "SAHP",
    "SelfCorrecting",
    "THP",
)
RTOL = 1e-5
ATOL = 1e-6


def digest_json(path):
    canonical = json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def compare(actual, expected, path="snapshot"):
    """Compare every field; paths/timestamps are excluded at capture, not here."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: changed field names")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: changed length")
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise AssertionError(f"{path}: nonfinite/non-numeric result {actual}")
        if not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError(
                f"{path}: {actual} != {expected} (rtol={RTOL}, atol={ATOL})"
            )
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def run_model(model_name, output):
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    import new_ltpp.models as models
    from new_ltpp.configs import RunnerConfig
    from new_ltpp.runners.runner_manager import RunnerManager
    from new_ltpp.shared_types import Batch

    if set(models.__all__) != set(MODELS):
        raise AssertionError(
            "Public model catalogue changed: extend fixtures and baselines"
        )
    torch.set_num_threads(1)
    raw = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    raw["model_id"] = raw["model_config"]["model_id"] = model_name
    raw["save_dir"] = str(output / "runs")
    for key in ("train_dir", "valid_dir", "test_dir"):
        raw["data_config"][key] = str(FIXTURES / "fixture.json")
    config = RunnerConfig.model_validate(raw)
    if (config.base_dir / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory for each suite invocation")
    # Replay the saved effective configuration through the public CLI.
    config_path = output / "effective.yaml"
    config.save_to_yaml_file(config_path)
    from typer.testing import CliRunner

    from scripts.cli import app

    os.environ["LTPP_RUN_ID"] = "integration"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--config",
            str(config_path),
            "--phase",
            "all",
            "--save-dir",
            str(output / "runs"),
        ],
    )
    print(result.output)
    if result.exit_code != 0:
        raise RuntimeError(
            f"CLI all failed for {model_name}: {result.exception}"
        ) from result.exception
    manifest_path = config.base_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for phase in ("train", "test", "predict"):
        assert manifest["phases"][phase]["status"] == "completed", phase
        assert (
            manifest["phases"][phase]["checkpoint"]["sha256"]
            == manifest["phases"]["train"]["checkpoint"]["sha256"]
        )
    checkpoint_path = manifest["phases"]["train"]["checkpoint"]["path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 0 and checkpoint["global_step"] == 2

    manager = RunnerManager(config, checkpoint_path=checkpoint_path)
    manager.setup_runner(enable_logging=False)
    model = manager.runner.model
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    manager.runner.datamodule.setup(stage="test")
    batch = next(iter(manager.runner.datamodule.test_dataloader()))
    batch = Batch(*(tensor[:2].clone() for tensor in batch.to_mapping().values()))
    torch.manual_seed(456)
    # FullyNN requires autograd even when evaluating intensities.
    loss, num_events = model.loglike_loss(batch)
    parameter_gradients = torch.autograd.grad(
        loss, tuple(model.parameters()), allow_unused=True
    )
    sample_dtimes = torch.full((*batch.time_seqs.shape, 2), 0.2)
    intensities = None
    if getattr(model, "supports_intensity", True):
        intensities = model.compute_intensities_at_sample_dtimes(
            time_seqs=batch.time_seqs,
            time_delta_seqs=batch.time_delta_seqs,
            type_seqs=batch.type_seqs,
            sample_dtimes=sample_dtimes,
            valid_event_mask=batch.valid_event_mask,
        )

    def tensor_data(tensor):
        if tensor is None:
            return None
        tensor = tensor.detach().cpu()
        assert torch.isfinite(tensor).all(), "nonfinite tensor"
        return {"shape": list(tensor.shape), "values": tensor.reshape(-1).tolist()}

    records = pq.read_table(
        config.base_dir / "simulations" / "sim_batch.parquet"
    ).to_pylist()
    assert len(records) == 40, "Expected eight complete simulations of five events"
    assert {row["seq_idx"] for row in records} == set(range(8))
    for sequence in range(8):
        events = [row for row in records if row["seq_idx"] == sequence]
        assert [row["event_idx"] for row in events] == list(range(5))
        assert all(row["type_event"] in (0, 1) for row in events)
        assert all(math.isfinite(row["time_since_start"]) for row in events)
        assert all(row["time_since_last_event"] >= 0 for row in events)
    numeric_csv = {}
    import pandas as pd

    results = pd.read_csv(config.base_dir / "results.csv")
    for key in results.select_dtypes(include=np.number).columns:
        if not key.startswith("Unnamed:"):
            numeric_csv[key] = [
                None if pd.isna(value) else value for value in results[key].tolist()
            ]
    plots = list(config.base_dir.rglob("*.png"))
    assert plots and all(path.stat().st_size > 0 for path in plots), "Missing plots"
    snapshot = {
        "schema": 1,
        "model": model_name,
        "fixture_sha256": digest_json(FIXTURES / "fixture.json"),
        "config_sha256": digest_json(FIXTURES / "config.json"),
        "path_preparation": manifest["numerics"]["signature_path_preparation"],
        "trained_state": {
            key: tensor_data(value) for key, value in checkpoint["state_dict"].items()
        },
        "probe_loss": tensor_data(loss),
        "probe_num_events": int(num_events),
        "probe_intensities": tensor_data(intensities),
        "probe_gradients": {
            name: tensor_data(gradient)
            for (name, _), gradient in zip(
                model.named_parameters(), parameter_gradients
            )
        },
        "test_metrics": json.loads(
            (config.base_dir / "test_results" / "test_results.json").read_text()
        ),
        "simulation_records": records,
        "simulation_numeric_results": numeric_csv,
    }
    # allow_nan=False also catches nonfinite values in JSON/CSV/Parquet metrics.
    write_json(output / "snapshot.json", snapshot)
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": manifest["environment"]["packages"],
            "code": manifest["code"],
            "lock_sha256": manifest["lock_sha256"],
            "rtol": RTOL,
            "atol": ATOL,
        },
    )
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(*MODELS, "all"),
        default=os.environ.get("INTEGRATION_MODEL", "all"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write candidates only; never overwrite references",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts" / "integration" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_model(args.model, output)
        return
    selected = MODELS if args.model == "all" else (args.model,)
    results = {}
    for model in selected:
        target = output / model
        target.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            MPLBACKEND="Agg",
            PYTHONUTF8="1",
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
        )
        env.pop("LTPP_RUN_ID", None)
        command = [
            sys.executable,
            "-m",
            "scripts.integration_suite",
            "--worker",
            "--model",
            model,
            "--output",
            str(target),
        ]
        try:
            with (target / "run.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                )
            if completed.returncode:
                raise AssertionError(
                    f"Pipeline failed (exit {completed.returncode}); see {target / 'run.log'}"
                )
            if not args.record:
                compare(
                    json.loads((target / "snapshot.json").read_text()),
                    json.loads((FIXTURES / "baselines" / f"{model}.json").read_text()),
                )
            results[model] = {"status": "candidate" if args.record else "passed"}
        except Exception as error:
            results[model] = {"status": "failed", "error": str(error)}
        print(f"{model}: {results[model]}", flush=True)
        write_json(output / "report.json", results)
    if any(value["status"] == "failed" for value in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
