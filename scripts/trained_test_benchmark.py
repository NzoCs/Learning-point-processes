"""Prepare/check or execute one explicit trained-model test evaluation.

Default is read-only preflight. No downloads, training, Slurm or artifact repair.
Checkpoint loading uses the current ModelFactory and a strict state-dict load;
do not load untrusted pickle checkpoints. Historical incompatibility requires
separate migration.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from new_ltpp.configs import ModelConfig
from new_ltpp.configs.window_test_config import WindowTestConfig
from new_ltpp.evaluation.benchmarks.trained_test_suite import evaluate_test_suite
from new_ltpp.evaluation.statistical_testing.observations import configurations
from new_ltpp.evaluation.statistical_testing.references import NeuralIntensityReference
from new_ltpp.models.model_factory import ModelFactory
from new_ltpp.shared_types import Batch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_pool(path):
    with np.load(path, allow_pickle=False) as data:
        times = torch.as_tensor(data["time_seqs"]).double()
        if (
            data["type_seqs"].dtype.kind not in "iu"
            or data["valid_event_mask"].dtype.kind != "b"
        ):
            raise ValueError("Snapshot marks must be integers and masks boolean")
        marks = torch.as_tensor(data["type_seqs"]).long()
        mask = torch.as_tensor(data["valid_event_mask"]).bool()
        ids = data["sequence_ids"].astype(str).tolist()
        horizon = float(data["horizon"])
        coverage = np.asarray(data["coverage_end"])
        if (
            coverage.shape != (len(times),)
            or not np.isfinite(coverage).all()
            or np.any(coverage < horizon)
        ):
            raise ValueError(
                "Every source trajectory must cover the full window; event-budget tails cannot be assumed empty"
            )
    if len(ids) != len(times) or len(ids) != len(set(ids)) or any(not v for v in ids):
        raise ValueError("Pools must contain distinct independent trajectory IDs")
    gaps = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1])).masked_fill(
        ~mask, 0
    )
    return Batch(times, gaps, marks, mask, sequence_ids=ids), horizon


def evaluate_manifest(path, *, execute=False):
    manifest = json.loads(Path(path).read_text())
    required = {
        "model_id",
        "checkpoint",
        "checkpoint_sha256",
        "observed",
        "observed_sha256",
        "simulated",
        "simulated_sha256",
        "training_ids",
        "training_ids_sha256",
        "tests",
        "simulation_checkpoint_sha256",
        "observation_protocol",
        "device",
        "output",
    }
    if set(manifest) != required:
        raise ValueError(
            f"Exactly these manifest fields are required: {sorted(required)}"
        )
    if manifest["observation_protocol"] != "independent_fixed_windows_v1":
        raise ValueError(
            "Export fixed windows with known exposure, not last-event horizons or dependent sliding windows"
        )
    if manifest["simulation_checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("Simulation must come from the same frozen checkpoint")
    if manifest["device"] not in ["cpu", "cuda"]:
        raise ValueError("Explicit cpu or cuda device required")
    files = {}
    for name in ["checkpoint", "observed", "simulated", "training_ids"]:
        file = Path(manifest[name])
        file = file if file.is_absolute() else Path(path).resolve().parent / file
        if not file.is_file() or digest(file) != manifest[name + "_sha256"]:
            raise ValueError(f"Missing or changed artifact: {name}")
        files[name] = file
    observed, h1 = load_pool(files["observed"])
    simulated, h2 = load_pool(files["simulated"])
    training_ids = json.loads(files["training_ids"].read_text())
    if not isinstance(training_ids, list):
        raise ValueError("training_ids must be a JSON list")
    if set(observed.sequence_ids) & set(training_ids):
        raise ValueError("Training/evaluation leakage")
    if set(observed.sequence_ids) & set(simulated.sequence_ids):
        raise ValueError("Observed and simulation IDs overlap")
    configs = [WindowTestConfig.model_validate(v) for v in manifest["tests"]]
    if not configs or h1 != h2 or any(c.horizon != h1 for c in configs):
        raise ValueError("All methods and pools require the same explicit horizon")
    if len({c.num_event_types for c in configs}) != 1:
        raise ValueError("All tests must share a mark alphabet")
    for c in configs:
        configurations(observed, c.horizon, c.num_event_types)
        configurations(simulated, c.horizon, c.num_event_types)
    if len(observed.time_seqs) != len(simulated.time_seqs):
        raise ValueError("Pool sizes differ")
    report = dict(
        status="artifact_preflight_passed",
        executed=execute,
        n=len(observed.time_seqs),
        checkpoint_sha256=manifest["checkpoint_sha256"],
        test_configs=[c.model_dump() for c in configs],
    )
    if not execute:
        return report
    if manifest["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by manifest, no CPU fallback")
    destination = Path(manifest["output"])
    if not destination.is_absolute():
        destination = Path(path).resolve().parent / destination
    if destination.exists():
        raise FileExistsError("Refuse to overwrite results")
    # Normal current architecture only. No import aliases, shape surgery, or retraining.
    import new_ltpp.models  # noqa: F401

    checkpoint = torch.load(files["checkpoint"], map_location="cpu", weights_only=False)
    config = ModelConfig.model_validate(checkpoint["hyper_parameters"]["model_config"])
    if config.model_id != manifest["model_id"]:
        raise ValueError("Checkpoint model identity differs from manifest")
    model = ModelFactory.create_model_by_name(
        manifest["model_id"],
        config,
        checkpoint["hyper_parameters"]["data_info"],
        str(destination.parent / "model"),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(manifest["device"]).eval()
    observed.to_device(torch.device(manifest["device"]))
    simulated.to_device(torch.device(manifest["device"]))
    reference = NeuralIntensityReference(model)
    if model.num_event_types != configs[0].num_event_types:
        raise ValueError("Checkpoint and dataset mark alphabets differ")
    if any(c.test_type in {"relative_loss", "ksd"} for c in configs):
        reference.intensities(
            torch.empty(0), torch.empty(0, dtype=torch.long), 0.0, np.array([h1 / 2])
        )
    report.update(evaluate_test_suite(observed, simulated, reference, configs))
    report["status"] = "complete"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Evaluate trusted checkpoints after read-only preflight",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate_manifest(args.manifest, execute=args.execute), indent=2))
