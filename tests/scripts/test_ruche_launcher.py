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


def test_invalid_epoch_override_fails_in_dry_run(tmp_path):
    result = launch(
        "run_all_pipeline.sh", tmp_path, ("--dry-run",), {"LTPP_EPOCHS": "0"}
    )
    assert result.returncode == 2
    assert "LTPP_EPOCHS" in result.stderr
