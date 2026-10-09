"""Persist effective configuration and provenance before executing a phase."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import torch
from new_ltpp.evaluation.statistical_testing.point_process_kernels.utils import (
    SIGNATURE_PATH_PREPARATION,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(root, args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


class RunManifest:
    def __init__(self, config):
        self.config = config
        self.path = config.base_dir / "manifest.json"
        raw_config = config.get_yaml_config()
        encoded = json.dumps(raw_config, sort_keys=True).encode()
        config_hash = hashlib.sha256(encoded).hexdigest()
        if self.path.exists():
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
            if self.value["config_sha256"] != config_hash:
                raise ValueError(
                    "Existing run has a different configuration; use a new run_id"
                )
            return
        root = Path(__file__).resolve().parents[2]
        dataset = config.data_config
        files = {}
        for split, source in (
            ("train", dataset.train_dir),
            ("dev", dataset.valid_dir),
            ("test", dataset.test_dir),
        ):
            if dataset.data_format != "hf" and Path(source).is_file():
                files[split] = {"path": source, "sha256": sha256_file(source)}
        self.value = {
            "schema_version": 1,
            "run_id": config.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": raw_config,
            "config_sha256": config_hash,
            "code": {
                "commit": _git_value(root, ["rev-parse", "HEAD"]),
                "dirty": _git_value(root, ["status", "--porcelain"]) not in (None, ""),
            },
            "lock_sha256": sha256_file(root / "uv.lock")
            if (root / "uv.lock").is_file()
            else None,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": sorted(
                    (d.metadata["Name"], d.version)
                    for d in distributions()
                    if d.metadata["Name"]
                ),
            },
            "numerics": {
                "torch_cuda": torch.version.cuda,
                "deterministic": config.training_config.deterministic,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "mmd_estimator": "unbiased_off_diagonal_v2",
                "signature_path_preparation": SIGNATURE_PATH_PREPARATION,
            },
            "data": {
                "format": dataset.data_format,
                "revision": dataset.revision,
                "local_files": files,
            },
            "phases": {},
        }
        config.save_to_yaml_file(config.base_dir / "effective_config.yaml")
        self.save()

    def save(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.value, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def start(self, phase):
        self.value["phases"][phase] = {"status": "running"}
        self.save()

    def finish(self, phase, checkpoint=None, error=None):
        result = {"status": "failed" if error is not None else "completed"}
        if error is not None:
            result["error"] = str(error)
        if checkpoint and Path(checkpoint).is_file():
            result["checkpoint"] = {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            }
        self.value["phases"][phase] = result
        self.value["numerics"]["float32_matmul_precision"] = (
            torch.get_float32_matmul_precision()
        )
        self.save()
