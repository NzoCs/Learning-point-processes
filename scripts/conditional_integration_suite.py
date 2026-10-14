"""Opt-in conditional CLI integration with independently generated train/dev/test data."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.integration_suite import FIXTURES, MODELS, ROOT, compare, write_json


def make_splits():
    splits = {}
    for split, seed in (("train", 1701), ("dev", 1702), ("test", 1703)):
        rng = random.Random(seed)
        records = []
        for index in range(8):
            gaps = [0.0] + [0.05 + rng.expovariate(4.0) for _ in range(4)]
            records.append(
                {
                    "sequence_id": f"conditional-{split}-{index}",
                    "dim_process": 2,
                    "time_since_start": [sum(gaps[: i + 1]) for i in range(5)],
                    "time_since_last_event": gaps,
                    "type_event": [rng.randrange(2) for _ in range(5)],
                }
            )
        splits[split] = records
    identifiers = [r["sequence_id"] for rows in splits.values() for r in rows]
    assert len(set(identifiers)) == 24
    paths = [tuple(r["time_since_start"]) for rows in splits.values() for r in rows]
    assert len(set(paths)) == 24
    return splits


def run_model(name, output, *, epochs=1, checkpoint_path=None):
    import pyarrow.parquet as pq
    import torch
    from typer.testing import CliRunner

    import new_ltpp.models as models
    from new_ltpp.configs import RunnerConfig
    from new_ltpp.evaluation.accumulators.summary_statistics_accumulator import (
        BatchStatisticsCollector,
    )
    from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
        monte_carlo_p_value,
    )
    from new_ltpp.models.base.training import TrainingMixin
    from new_ltpp.models.simulation.tpp_io import SimulationIOManager
    from new_ltpp.runners.run_manifest import sha256_file
    from scripts.cli import app

    assert set(models.__all__) == set(MODELS), "Update the tested model catalogue"
    torch.set_num_threads(1)
    splits = make_splits()
    raw = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    raw["model_id"] = raw["model_config"]["model_id"] = name
    raw["save_dir"] = str(output / "runs")
    raw["training_config"]["max_epochs"] = epochs
    for split, key in (
        ("train", "train_dir"),
        ("dev", "valid_dir"),
        ("test", "test_dir"),
    ):
        path = output / "data" / f"{split}.json"
        write_json(path, splits[split])
        raw["data_config"][key] = str(path)
    raw["simulation_config"]["conditional_evaluation"] = {
        "history_events": 2,
        "future_events": 3,
    }
    raw["statistical_test_config"]["n_samples"] = 99
    raw["run_id"] = "conditional-integration"
    config = RunnerConfig.model_validate(raw)
    if (config.base_dir / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory")
    config_path = output / "effective.yaml"
    config.save_to_yaml_file(config_path)
    expected = {
        "provided:" + json.dumps(row["sequence_id"]): row for row in splits["test"]
    }
    exported = []
    simulated_ids = []
    captured = []
    prediction_calls = []
    predict = TrainingMixin.predict_step
    update = BatchStatisticsCollector.update
    compute = BatchStatisticsCollector.compute

    def guarded_predict(model, batch, batch_idx, **kwargs):
        assert not model.training, "Conditional model must be frozen in eval mode"
        state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        simulate = model.simulate

        def prefix_only(history, num_events_to_simulate):
            assert history.time_seqs.shape == (4, 2)
            assert num_events_to_simulate == 3
            assert history.sequence_ids == batch.sequence_ids
            for index, identifier in enumerate(history.sequence_ids):
                torch.testing.assert_close(
                    history.time_seqs[index],
                    torch.tensor(expected[identifier]["time_since_start"][:2]),
                )
            prediction_calls.append(list(history.sequence_ids))
            return simulate(history, num_events_to_simulate=num_events_to_simulate)

        with patch.object(model, "simulate", side_effect=prefix_only):
            result = predict(model, batch, batch_idx, **kwargs)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        return result

    def checked_update(collector, future, simulation, *, histories=None, pair_ids=None):
        assert histories is not None and future.sequence_ids == pair_ids
        assert simulation.sequence_ids == pair_ids
        assert future.time_seqs.shape == simulation.time_seqs.shape == (4, 3)
        assert simulation.valid_event_mask.all()
        assert torch.isfinite(simulation.time_seqs).all()
        assert (simulation.time_seqs >= 0).all()
        assert (torch.diff(simulation.time_seqs, dim=1) >= 0).all()
        assert ((simulation.type_seqs >= 0) & (simulation.type_seqs < 2)).all()
        for index, identifier in enumerate(pair_ids):
            source = torch.tensor(expected[identifier]["time_since_start"])
            torch.testing.assert_close(future.time_seqs[index], source[2:] - source[1])
            torch.testing.assert_close(histories.time_seqs[index], source[:2])
        simulated_ids.extend(pair_ids)
        exported.extend(
            SimulationIOManager.format_simulation_result(
                simulation, 2, len(simulated_ids) - 4, preserve_origin=True
            )
        )
        return update(
            collector, future, simulation, histories=histories, pair_ids=pair_ids
        )

    def capture_compute(collector):
        stats = compute(collector)
        captured.append(
            (
                stats["statistical_tests"],
                set(collector._accumulators["statistical_tests"].test._seen_pair_ids),
            )
        )
        return stats

    os.environ["LTPP_RUN_ID"] = config.run_id
    command = [
        "run",
        "--config",
        str(config_path),
        "--phase",
        "all",
        "--save-dir",
        str(output / "runs"),
    ]
    if checkpoint_path is not None:
        command.extend(["--checkpoint", str(checkpoint_path)])
    with (
        patch.object(TrainingMixin, "predict_step", guarded_predict),
        patch.object(BatchStatisticsCollector, "update", checked_update),
        patch.object(BatchStatisticsCollector, "compute", capture_compute),
    ):
        result = CliRunner().invoke(
            app,
            command,
        )
    print(result.output)
    if result.exit_code:
        raise RuntimeError(
            f"CLI all failed for {name}: {result.exception}"
        ) from result.exception
    manifest = json.loads(
        (config.base_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config"]["model_id"] == name
    checkpoint = manifest["phases"]["train"]["checkpoint"]["sha256"]
    for phase in ("train", "test", "predict"):
        assert manifest["phases"][phase]["status"] == "completed"
        assert manifest["phases"][phase]["checkpoint"]["sha256"] == checkpoint
    hashes = {
        split: manifest["data"]["local_files"][split]["sha256"] for split in splits
    }
    assert len(set(hashes.values())) == 3
    files = list(config.base_dir.rglob("*.parquet"))
    assert len(files) == 1
    events = pq.read_table(files[0]).to_pylist()
    compare(events, exported)
    assert len(events) == 24 and len(prediction_calls) == 2
    assert len(simulated_ids) == len(set(simulated_ids)) == 8
    assert set(simulated_ids) == set(expected)
    stats, seen_ids = captured[-1]
    assert seen_ids == set(expected) and stats["num_sequences"] == 8
    assert stats["null_methods"] == ["within_history_pair_swaps_joint_kernel_v1"] * 2
    assert stats["null_sample_counts"] == [99, 99]
    assert all(math.isfinite(p) and 0 < p <= 1 for p in stats["p_values"])
    observed = torch.tensor(stats["observed_statistic"], dtype=torch.float64).sum()
    null = (
        torch.tensor(stats["permuted_statistic"], dtype=torch.float64)
        .reshape(2, 99)
        .sum(0)
    )
    assert math.isfinite(observed) and torch.isfinite(null).all()
    assert math.isclose(
        stats["pooled_p_value"],
        monte_carlo_p_value(observed, null).item(),
        abs_tol=1e-12,
    )
    write_json(
        output / "checks.json",
        {
            "model": name,
            "status": "passed",
            "code": manifest["code"],
            "environment": manifest["environment"],
            "lock_sha256": manifest["lock_sha256"],
            "data_sha256": hashes,
            "checkpoint_sha256": checkpoint,
            "fit_evaluation_disjoint": True,
            "independent_generation_seeds": [1701, 1702, 1703],
            "model_frozen": True,
            "prefix_only_simulation": True,
            "history_events": 2,
            "future_events": 3,
            "batches": 2,
            "sequence_ids": sorted(seen_ids),
            "export_sha256": sha256_file(files[0]),
            "exported_events": len(events),
            "statistical_tests": stats,
            "scope": "software_integration_not_calibration_or_model_correctness",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", *MODELS),
        default=os.environ.get("INTEGRATION_MODEL", "all"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_model(args.model, output)
        return
    results = {}
    for name in MODELS if args.model == "all" else (args.model,):
        target = output / name
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
        try:
            with (target / "run.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.conditional_integration_suite",
                        "--worker",
                        "--model",
                        name,
                        "--output",
                        str(target),
                    ],
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
            results[name] = {"status": "passed"}
        except Exception as error:
            results[name] = {"status": "failed", "error": str(error)}
        print(f"{name}: {results[name]}", flush=True)
        write_json(output / "report.json", results)
    if any(value["status"] == "failed" for value in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
