"""Execute every Python cell in maintained notebooks, offline in a fresh process.

No Jupyter server is required. Outputs and per-file hashes go to artifacts.
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "NewLTPP_Getting_Started.ipynb",
    "Hawkes_MMD_Test.ipynb",
    "Test_MMD_Metric.ipynb",
)


def execute(path, output):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    started, count = time.monotonic(), 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{path.stem}.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    print(f"Executing cell {index}", flush=True)
                    exec(
                        compile(cell.source, f"{path.name}:cell{index}", "exec"),
                        namespace,
                    )
                    count += 1
    report = {
        "notebook": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "executed_code_cells": count,
        "duration_seconds": time.monotonic() - started,
        "python": sys.version,
        "status": "passed",
        "execution": "all Python cells, shared namespace, fresh subprocess",
    }
    (output / f"{path.stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/notebook-validation"
    )
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["MPLBACKEND"] = "Agg"
    if args.worker:
        execute(args.worker, args.output)
        return
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        PYTHONUTF8="1",
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    for name in NOTEBOOKS:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.validate_notebooks",
                "--worker",
                str(ROOT / "notebooks" / name),
                "--output",
                str(args.output.resolve()),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        print(f"Passed: {name}", flush=True)


if __name__ == "__main__":
    main()
