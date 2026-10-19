"""Exercise launch plans without installing native dependencies or reserving GPUs."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
if os.name == "nt":
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.exists():
        BASH = str(git_bash)
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required")


def launch(script, cwd, args=(), env=None):
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "LTPP_"))
    }
    clean_env.update(env or {})
    return subprocess.run(
        [BASH, str(ROOT / "scripts/bash" / script), *args],
        cwd=cwd,
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_all_event_replay_plan_preserves_both_original_archives(tmp_path):
    env = {
        "LTPP_MMD_SOURCE": str(tmp_path / "mmd archive"),
        "LTPP_LANDMARK_SOURCE": str(tmp_path / "landmark archive"),
        "LTPP_ALL_EVENT_OUTPUT": str(tmp_path / "new output"),
    }
    result = launch("ruche_all_event_rmse.sh", tmp_path, ("--dry-run",), env)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 20
    for command in commands:
        assert command[command.index("--mode") + 1] == "all_events"
        assert command[command.index("--device") + 1] == "cuda"
        assert "landmark archive" in command[command.index("--landmark-source") + 1]
        assert any("mmd archive" in argument for argument in command)
        assert any("new output" in argument for argument in command)
    assert not list(tmp_path.iterdir())


def test_scientific_campaign_plan_keeps_independent_cohorts_and_no_execution(tmp_path):
    output = tmp_path / "results with spaces"
    result = launch(
        "ruche_report_experiments.sh",
        tmp_path,
        ("--dry-run",),
        env={"LTPP_SCIENCE_OUTPUT": str(output)},
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 20
    groups = {}
    for command in commands:
        case = command[command.index("--case") + 1]
        cohort = command[command.index("--cohort") + 1]
        groups.setdefault(case, []).append(cohort)
        assert command[command.index("--trials") + 1] == (
            "1000" if case.endswith("H0") else "250"
        )
        assert command[command.index("--device") + 1] == "cuda"
        assert command[command.index("--n") + 1] == "128"
        assert command[command.index("--permutations") + 1] == "499"
        assert "results with spaces" in command[command.index("--output") + 1]
    assert len(groups) == 10 and sum(case.endswith("H0") for case in groups) == 4
    assert all(sorted(cohorts) == ["0", "1"] for cohorts in groups.values())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("index", ["20", "-1", "abc"])
def test_scientific_campaign_rejects_bad_array_before_python(index, tmp_path):
    result = launch(
        "ruche_report_experiments.sh",
        tmp_path,
        env={"SLURM_JOB_ID": "42", "SLURM_ARRAY_TASK_ID": index},
    )
    assert result.returncode == 2
    assert "Slurm array" in result.stderr
    assert not result.stdout and not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "script,count",
    [("run_all_pipeline.sh", 3), ("train_ruche_cpu.sh", 28), ("smoke_ruche_gpu.sh", 1)],
)
def test_dry_run_from_another_directory_uses_valid_grid(script, count, tmp_path):
    result = launch(script, tmp_path, ("--dry-run",))
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == count
    presets = yaml.safe_load((ROOT / "yaml_configs/configs.yaml").read_text())
    pairs = set()
    for index, command in enumerate(commands):
        assert command[0] == f"task={index}"
        dataset = command[command.index("--dataset-id") + 1]
        model = command[command.index("--model") + 1]
        assert dataset in presets["data_configs"]
        assert "--data-config" not in command
        assert "--gpu" not in command
        pairs.add((model, dataset))
    assert len(pairs) == count
    assert not list(tmp_path.iterdir()), "Dry-run must not create artifacts"
    if script == "smoke_ruche_gpu.sh":
        command = commands[0]
        assert command[command.index("--phase") + 1] == "train"
        assert command[command.index("--epochs") + 1] == "1"


@pytest.mark.parametrize("index", ["3", "-1", "abc", "999999999999999999999"])
def test_bad_gpu_index_fails_before_launch(index, tmp_path):
    result = launch("run_all_pipeline.sh", tmp_path, env={"SLURM_ARRAY_TASK_ID": index})
    assert result.returncode == 2
    assert "index" in result.stderr
    assert not result.stdout


def test_refuses_training_on_login_node(tmp_path):
    result = launch("run_all_pipeline.sh", tmp_path)
    assert result.returncode == 2
    assert "sbatch" in result.stderr


def test_learned_rmse_plan_uses_frozen_training_and_known_archive(tmp_path):
    result = launch(
        "ruche_learned_rmse.sh",
        tmp_path,
        ("--dry-run",),
        env={
            "LTPP_KNOWN_RMSE_SOURCE": "/tmp/known cells",
            "LTPP_LEARNED_TRAINING": "/tmp/frozen training",
            "LTPP_LEARNED_OUTPUT": "/tmp/new results",
        },
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 20
    for i, command in enumerate(commands):
        assert command[0] == f"task={i}"
        assert "scripts.report_learned_rmse" in command
        assert "/tmp/frozen training" in command
        assert command[-2:] == ["--device", "cuda"]
        assert command[command.index("replay") + 1].endswith(f"/cohort-{i%2}")
    assert not list(tmp_path.iterdir())


def test_invalid_epoch_override_fails_in_dry_run(tmp_path):
    result = launch(
        "run_all_pipeline.sh", tmp_path, ("--dry-run",), {"LTPP_EPOCHS": "0"}
    )
    assert result.returncode == 2
    assert "LTPP_EPOCHS" in result.stderr
