"""Artifact preflight is read-only; observed exposure and split leakage are gates."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from scripts.independent_simulator_validation import create_model
from scripts.trained_test_benchmark import evaluate_manifest


def make_manifest(tmp_path, *, coverage=1.0, overlap=False):
    for name, ids in [
        ("observed", ["test-a", "test-b"]),
        ("simulated", ["sim-a", "sim-b"]),
    ]:
        np.savez(
            tmp_path / f"{name}.npz",
            time_seqs=np.array([[0.2], [0.4]]),
            type_seqs=np.zeros((2, 1), dtype=np.int64),
            valid_event_mask=np.ones((2, 1), dtype=bool),
            horizon=np.array(1.0),
            coverage_end=np.full(2, coverage),
            sequence_ids=np.array(ids),
        )
    (tmp_path / "weights.ckpt").write_bytes(
        b"not-a-checkpoint-no-loading-during-preflight"
    )
    (tmp_path / "training.json").write_text(
        json.dumps(["test-a"] if overlap else ["train-a"])
    )
    manifest = dict(
        model_id="NHP",
        checkpoint="weights.ckpt",
        observed="observed.npz",
        simulated="simulated.npz",
        training_ids="training.json",
        device="cpu",
        output=str(tmp_path / "result.json"),
        observation_protocol="independent_fixed_windows_v1",
        tests=[dict(test_type="window_mmd", horizon=1, num_event_types=1, n_samples=7)],
    )
    for name in ["checkpoint", "observed", "simulated", "training_ids"]:
        manifest[name + "_sha256"] = hashlib.sha256(
            (tmp_path / manifest[name]).read_bytes()
        ).hexdigest()
    manifest["simulation_checkpoint_sha256"] = manifest["checkpoint_sha256"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_preflight_never_loads_checkpoint_or_writes_results(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not unpickle weights")

    monkeypatch.setattr("torch.load", forbidden)
    result = evaluate_manifest(make_manifest(tmp_path))
    assert not result["executed"] and result["n"] == 2
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "coverage,overlap,message", [(0.5, False, "cover"), (1.0, True, "leakage")]
)
def test_preflight_rejects_unknown_tail_and_fit_evaluation_leakage(
    tmp_path, coverage, overlap, message
):
    with pytest.raises(ValueError, match=message):
        evaluate_manifest(make_manifest(tmp_path, coverage=coverage, overlap=overlap))


def test_current_trained_checkpoint_loads_through_public_evaluation_command(tmp_path):
    manifest_path = make_manifest(tmp_path)
    model = create_model("NHP", tmp_path)
    raw = json.loads(
        (Path(__file__).parent / "integration/baselines/NHP.json").read_text()
    )
    model.load_state_dict(
        {
            k: torch.tensor(v["values"]).reshape(v["shape"])
            for k, v in raw["trained_state"].items()
        }
    )
    torch.save(
        dict(
            state_dict=model.state_dict(),
            hyper_parameters=dict(model.hparams),
            **{"pytorch-lightning_version": pl.__version__},
        ),
        tmp_path / "weights.ckpt",
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_sha256"] = hashlib.sha256(
        (tmp_path / "weights.ckpt").read_bytes()
    ).hexdigest()
    manifest["simulation_checkpoint_sha256"] = manifest["checkpoint_sha256"]
    manifest["tests"][0].update(num_event_types=2, grid=8)
    manifest_path.write_text(json.dumps(manifest))
    report = evaluate_manifest(manifest_path, execute=True)
    assert report["status"] == "complete" and report["executed"]
    assert (
        len(report["tests"]) == 1
        and len(report["tests"][0]["permuted_statistics"]) == 7
    )
    assert (tmp_path / "result.json").exists()
    with pytest.raises(FileExistsError):
        evaluate_manifest(manifest_path, execute=True)
