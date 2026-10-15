"""Exact CPU checks for conditional repeats, checkpoint replay and epoch-boundary resume."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from scripts.conditional_integration_suite import run_model
from scripts.integration_suite import CATALOGUE, write_json


def exact(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert left == right, (left, right)


def run(name, output):
    import pyarrow.parquet as pq
    import torch
    from typer.testing import CliRunner

    from new_ltpp.configs import RunnerConfig
    from new_ltpp.evaluation.accumulators.summary_statistics_accumulator import (
        BatchStatisticsCollector,
    )
    from scripts.cli import app

    torch.set_num_threads(1)
    paths = {
        key: output / key
        for key in ("first", "repeat", "continuous", "resumed", "replay")
    }
    for key in ("first", "repeat", "continuous"):
        run_model(name, paths[key], epochs=2 if key == "continuous" else 1)

    def manifest(key):
        files = list(paths[key].rglob("manifest.json"))
        assert len(files) == 1
        return json.loads(files[0].read_text()), files[0].parent

    def state(key, filename="last.ckpt"):
        _, root = manifest(key)
        return torch.load(
            root / "checkpoints" / filename, map_location="cpu", weights_only=False
        )

    def events(key):
        files = list(paths[key].rglob("*.parquet"))
        assert len(files) == 1
        return pq.read_table(files[0]).to_pylist()

    def stats(key):
        return json.loads((paths[key] / "checks.json").read_text())["statistical_tests"]

    exact(state("first")["state_dict"], state("repeat")["state_dict"])
    exact(events("first"), events("repeat"))
    exact(stats("first"), stats("repeat"))
    first, root = manifest("first")
    run_model(
        name,
        paths["resumed"],
        epochs=2,
        checkpoint_path=root / "checkpoints" / "last.ckpt",
    )
    for key in (
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "epoch",
        "global_step",
    ):
        exact(state("continuous")[key], state("resumed")[key])
    assert state("continuous")["global_step"] == 4
    exact(
        state("continuous", "best.ckpt")["state_dict"],
        state("resumed", "best.ckpt")["state_dict"],
    )
    exact(events("continuous"), events("resumed"))
    exact(stats("continuous"), stats("resumed"))

    raw = json.loads((root / "manifest.json").read_text())["config"]
    raw["save_dir"] = str(paths["replay"] / "runs")
    config = RunnerConfig.model_validate(raw)
    config_path = paths["replay"] / "effective.yaml"
    config.save_to_yaml_file(config_path)
    captured = []
    compute = BatchStatisticsCollector.compute

    def capture(collector):
        result = compute(collector)
        captured.append(result["statistical_tests"])
        return result

    os.environ["LTPP_RUN_ID"] = config.run_id
    with patch.object(BatchStatisticsCollector, "compute", capture):
        result = CliRunner().invoke(
            app,
            [
                "run",
                "--config",
                str(config_path),
                "--phase",
                "predict",
                "--save-dir",
                str(paths["replay"] / "runs"),
                "--checkpoint",
                first["phases"]["train"]["checkpoint"]["path"],
            ],
        )
    print(result.output)
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    exact(events("first"), events("replay"))
    exact(stats("first"), captured[-1])
    replay, _ = manifest("replay")
    assert set(replay["phases"]) == {"predict"}
    assert (
        replay["phases"]["predict"]["checkpoint"]["sha256"]
        == first["phases"]["train"]["checkpoint"]["sha256"]
    )
    # Interrupt after one exported batch. The partial run must remain failed;
    # restart the whole prediction in a fresh output, from the same checkpoint.
    from new_ltpp.models.base.training import TrainingStepsMixin

    interrupted = output / "interrupted-prediction"
    interrupted_raw = config.get_yaml_config()
    interrupted_raw["save_dir"] = str(interrupted / "runs")
    interrupted_config = RunnerConfig.model_validate(interrupted_raw)
    interrupted_path = interrupted / "effective.yaml"
    interrupted_path.parent.mkdir(parents=True, exist_ok=True)
    interrupted_config.save_to_yaml_file(interrupted_path)
    original_predict = TrainingStepsMixin.predict_step

    def interrupt_prediction(model, batch, batch_idx, **kwargs):
        if batch_idx == 1:
            raise RuntimeError("injected mid-prediction interruption")
        return original_predict(model, batch, batch_idx, **kwargs)

    with patch.object(TrainingStepsMixin, "predict_step", interrupt_prediction):
        interrupted_result = CliRunner().invoke(
            app,
            [
                "run",
                "--config",
                str(interrupted_path),
                "--phase",
                "predict",
                "--save-dir",
                str(interrupted / "runs"),
                "--checkpoint",
                first["phases"]["train"]["checkpoint"]["path"],
            ],
        )
    assert interrupted_result.exit_code != 0
    interrupted_manifest = list(interrupted.rglob("manifest.json"))
    assert len(interrupted_manifest) == 1
    partial = json.loads(interrupted_manifest[0].read_text())
    assert partial["phases"]["predict"]["status"] == "failed"
    restart = output / "restarted-prediction"
    restart_raw = config.get_yaml_config()
    restart_raw["save_dir"] = str(restart / "runs")
    restart_config = RunnerConfig.model_validate(restart_raw)
    restart_path = restart / "effective.yaml"
    restart_path.parent.mkdir(parents=True, exist_ok=True)
    restart_config.save_to_yaml_file(restart_path)
    captured.clear()
    with patch.object(BatchStatisticsCollector, "compute", capture):
        restarted_result = CliRunner().invoke(
            app,
            [
                "run",
                "--config",
                str(restart_path),
                "--phase",
                "predict",
                "--save-dir",
                str(restart / "runs"),
                "--checkpoint",
                first["phases"]["train"]["checkpoint"]["path"],
            ],
        )
    assert restarted_result.exit_code == 0, restarted_result.exception
    restart_parquet = list(restart.rglob("*.parquet"))
    assert len(restart_parquet) == 1
    exact(events("first"), pq.read_table(restart_parquet[0]).to_pylist())
    exact(stats("first"), captured[-1])
    report = dict(
        model=name,
        status="passed",
        code=first["code"],
        environment=first["environment"],
        lock_sha256=first["lock_sha256"],
        same_seed_repeat="exact",
        checkpoint_prediction_replay="exact",
        epoch_boundary_resume="exact",
        selected_best_checkpoint="exact",
        optimizer_and_scheduler="exact",
        interrupted_prediction_status="failed",
        whole_prediction_restart_after_interruption="exact",
        completed_steps=4,
        compared_fields=[
            "trained_weights",
            "optimizer",
            "scheduler",
            "selected_best_weights",
            "exported_events",
            "observed_statistics",
            "all_null_draws",
            "batch_and_aggregated_p_values",
        ],
        scope="CPU single process, zero loader workers; epoch-boundary training resume, no mid-prediction resume",
    )
    write_json(output / "report.json", report)
    print(f"{name}: repeat/replay/resume exact", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CATALOGUE, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.output.resolve())


if __name__ == "__main__":
    main()
