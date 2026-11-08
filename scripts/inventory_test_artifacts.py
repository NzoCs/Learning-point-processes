"""Read-only, exhaustive local weight-artifact inventory for the supplied roots.

No unpickling, compatibility aliases, migration, downloads or checkpoint edits.
Raw data NPZ/PKL are counted separately; paths and byte hashes identify weights.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def inventory(roots):
    found = []
    data = []
    seen = set()
    for root in roots:
        for directory, dirs, files in os.walk(Path(root).resolve()):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", ".venv", "node_modules", "__pycache__"}
            ]
            for name in files:
                path = Path(directory) / name
                if path in seen:
                    continue
                seen.add(path)
                if path.suffix.lower() in {".npz", ".pkl", ".parquet", ".arrow"}:
                    data.append(str(path))
                    continue
                weight = path.suffix.lower() in {".ckpt", ".pt", ".pth", ".safetensors"}
                if (
                    not weight
                    and path.suffix == ".json"
                    and ("trained_states" in str(path) or "baselines" in path.parts)
                ):
                    try:
                        weight = "trained_state" in json.loads(path.read_text())
                    except (ValueError, UnicodeError):
                        weight = False
                if not weight:
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(block)
                found.append(
                    dict(
                        path=str(path),
                        bytes=path.stat().st_size,
                        sha256=digest.hexdigest(),
                    )
                )
    return dict(
        scope=[str(Path(r).resolve()) for r in roots],
        weights=found,
        data_artifact_paths=data,
        remote_inventory="Not available while Ruche SSH is unavailable; must be completed before reuse or migration",
        migration="This additive evaluation change does not change model class paths, weights or serialization. No artifacts rewritten or migrated.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("roots", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = inventory(a.roots)
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(
        json.dumps(
            dict(
                weight_artifacts=len(report["weights"]),
                data_artifacts=len(report["data_artifact_paths"]),
            )
        )
    )
