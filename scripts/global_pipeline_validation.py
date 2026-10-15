"""Opt-in same-machine before/after integration of global conditional MMD."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWED_PARENT = "5b41f13fb4eee9abba4754df45959cb5a1b0c46a"


def capture(name, output):
    import pyarrow.parquet as pq
    import torch

    from scripts.conditional_integration_suite import run_model as conditional
    from scripts.integration_suite import run_model, write_json

    output.mkdir(parents=True, exist_ok=True)
    run_model(name, output / "ordinary", allow_terminated=name in ("FullyNN", "RMTPP"))
    cases = ["conditional"] + (
        ["known-terminal"] if name in ("FullyNN", "RMTPP") else []
    )
    for case in cases:
        target = output / case
        conditional(name, target, terminal_fixture=case == "known-terminal")
        manifests = list(target.rglob("manifest.json"))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())
        checkpoint = torch.load(
            manifest["phases"]["train"]["checkpoint"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        torch.save(
            {
                k: checkpoint[k]
                for k in (
                    "state_dict",
                    "optimizer_states",
                    "lr_schedulers",
                    "epoch",
                    "global_step",
                )
            },
            target / "numerical_checkpoint.pt",
        )
        run = manifests[0].parent
        test_metrics = json.loads(
            (run / "test_results" / "test_results.json").read_text()
        )
        events = pq.read_table(next(run.rglob("*.parquet"))).to_pylist()
        statuses = (
            [
                json.loads(line)
                for line in (run / "simulations" / "sequence_status.jsonl")
                .read_text()
                .splitlines()
            ]
            if name in ("FullyNN", "RMTPP")
            else None
        )
        write_json(
            target / "numerical_snapshot.json",
            dict(test_metrics=test_metrics, events=events, terminal_status=statuses),
        )


def validate(name, output, reference):
    import torch

    from scripts.conditional_reproducibility import exact
    from scripts.conditional_reproducibility import run as repeat
    from scripts.integration_suite import compare, write_json

    capture(name, output)
    # Ordinary losses, predictions, gradients, simulations and p-values all stay fixed.
    before = json.loads((reference / "ordinary" / "snapshot.json").read_text())
    after = json.loads((output / "ordinary" / "snapshot.json").read_text())
    exact(before, after)
    report = dict(
        model=name,
        reviewed_parent=REVIEWED_PARENT,
        ordinary="all_numerical_fields_exact",
        conditional={},
    )
    for case in ("conditional", "known-terminal"):
        if not (output / case).exists():
            continue
        a, b = output / case, reference / case
        exact(
            torch.load(a / "numerical_checkpoint.pt", weights_only=False),
            torch.load(b / "numerical_checkpoint.pt", weights_only=False),
        )
        exact(
            json.loads((a / "numerical_snapshot.json").read_text()),
            json.loads((b / "numerical_snapshot.json").read_text()),
        )
        old = json.loads((b / "checks.json").read_text())["statistical_tests"]
        new = json.loads((a / "checks.json").read_text())["statistical_tests"]
        # Only rounding may differ in the historical batch diagnostics after blocking.
        for key in (
            "p_values",
            "observed_statistic",
            "permuted_statistic",
            "num_sequences",
            "null_sample_counts",
            "null_methods",
        ):
            compare(new[key], old[key], key)
        assert new["global_pair_ids"] == sorted(
            json.loads((a / "checks.json").read_text())["sequence_ids"]
        )
        report["conditional"][case] = dict(
            weights_optimizer_losses_predictions_simulations="exact",
            historical_batch_diagnostics="unchanged_with_float_tolerance",
            before_p_value=old["pooled_p_value"],
            after_p_value=new["pooled_p_value"],
            global_observed=new["global_observed_statistic"],
            pairs=len(new["global_pair_ids"]),
        )
    repeat(name, output / "reproducibility")
    report["repeat_checkpoint_replay_epoch_resume_and_restart"] = "exact"
    report["scope"] = (
        "CPU software integration; independent train/dev/test data; not a scientific calibration of all model parameters"
    )
    write_json(output / "validation.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--model", default="all")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    output = args.output.resolve()
    if args.reference_dir is not None:
        args.reference_dir = args.reference_dir.resolve()
    sys.path.insert(0, str(code_root))
    os.chdir(code_root)
    import new_ltpp
    from scripts.integration_suite import CATALOGUE, write_json

    assert Path(new_ltpp.__file__).resolve().is_relative_to(code_root)
    if args.worker:
        if args.capture:
            capture(args.model, output)
        else:
            validate(args.model, output, args.reference_dir.resolve())
        return
    results = {}
    for name in CATALOGUE if args.model == "all" else (args.model,):
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-X",
            "utf8",
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            name,
            "--output",
            str(target),
            "--code-root",
            str(code_root),
        ]
        if args.capture:
            command.append("--capture")
        else:
            if args.reference_dir is None:
                parser.error("--reference-dir is required for validation")
            command += ["--reference-dir", str(args.reference_dir.resolve() / name)]
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
        with (target / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=code_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        results[name] = dict(
            status="passed" if process.returncode == 0 else "failed",
            log=str(target / "run.log"),
        )
        if process.returncode == 0 and not args.capture:
            results[name].update(json.loads((target / "validation.json").read_text()))
        write_json(output / "report.json", results)
        print(name, results[name]["status"], flush=True)
    if any(r["status"] != "passed" for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
