"""Integration tests for CLI runners – side-effect verification.

Convention
----------
- Tests are plain functions prefixed with test_.
- `tmp_path` (built-in pytest fixture) provides an isolated, auto-cleaned directory.
- Each test documents the filesystem side effects it expects, then asserts them.

Side effects per command
------------------------
generate
    {tmp_path}/train.json
    {tmp_path}/dev.json
    {tmp_path}/test.json
    {tmp_path}/generation_metadata.json   ← JSON with generation_config + stats

info --output <file>
    <file>                                ← JSON with keys: timestamp, system_info

version, benchmark --list
    (no filesystem side effects)
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from scripts.cli import app

PROJECT_ROOT = Path(__file__).parent.parent.parent
runner = CliRunner()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_hawkes_creates_json_files(tmp_path):
    """
    Side effects:
        tmp_path/train.json
        tmp_path/dev.json
        tmp_path/test.json
        tmp_path/generation_metadata.json
    """
    result = runner.invoke(
        app,
        [
            "generate",
            "--method",
            "hawkes",
            "--num-sim",
            "50",
            "--dim",
            "2",
            "--output",
            str(tmp_path),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "train.json").exists()
    assert (tmp_path / "dev.json").exists()
    assert (tmp_path / "test.json").exists()
    assert (tmp_path / "generation_metadata.json").exists()


def test_generate_hawkes_metadata_content(tmp_path):
    """generation_metadata.json must contain valid generation_config."""
    runner.invoke(
        app,
        [
            "generate",
            "--method",
            "hawkes",
            "--num-sim",
            "50",
            "--dim",
            "2",
            "--output",
            str(tmp_path),
            "--seed",
            "42",
        ],
    )

    meta = json.loads((tmp_path / "generation_metadata.json").read_text())
    assert meta["generation_config"]["generation_method"] == "hawkes"
    assert meta["generation_config"]["num_simulations"] == 50


def test_generate_self_correcting_creates_json_files(tmp_path):
    """
    Side effects:
        tmp_path/train.json, dev.json, test.json, generation_metadata.json
    """
    result = runner.invoke(
        app,
        [
            "generate",
            "--method",
            "self_correcting",
            "--num-sim",
            "30",
            "--dim",
            "2",
            "--output",
            str(tmp_path),
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    for fname in ("train.json", "dev.json", "test.json", "generation_metadata.json"):
        assert (tmp_path / fname).exists(), f"{fname} missing"


def test_generate_unknown_method_exits_nonzero_writes_nothing(tmp_path):
    """Bad method → non-zero exit, no data files written."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--method",
            "does_not_exist",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "train.json").exists()


def test_generate_bad_splits_exits_nonzero(tmp_path):
    """Splits summing to > 1.0 → rejected before any file is written."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--train-ratio",
            "0.7",
            "--test-ratio",
            "0.7",
            "--dev-ratio",
            "0.2",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def test_info_without_output_exits_zero():
    """No --output: command succeeds, no file written."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0


def test_info_with_output_creates_json(tmp_path):
    """
    Side effect:
        tmp_path/system_info.json  ← JSON with timestamp + system_info
    """
    out_file = tmp_path / "system_info.json"

    result = runner.invoke(app, ["info", "--output", str(out_file)])

    assert result.exit_code == 0, result.output
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert "timestamp" in data
    assert "system_info" in data
    assert "system" in data["system_info"]


def test_info_output_contains_python_section(tmp_path):
    """The saved JSON must include a python section."""
    out_file = tmp_path / "info.json"
    runner.invoke(app, ["info", "--output", str(out_file)])

    data = json.loads(out_file.read_text())
    assert "python" in data["system_info"]


# ---------------------------------------------------------------------------
# version  (no filesystem side effects)
# ---------------------------------------------------------------------------


def test_version_exits_zero():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_version_output_mentions_cli():
    result = runner.invoke(app, ["version"])
    assert "EasyTPP CLI v4.0" in result.output


# ---------------------------------------------------------------------------
# benchmark --list  (no filesystem side effects)
# ---------------------------------------------------------------------------


def test_benchmark_list_exits_zero():
    result = runner.invoke(app, ["benchmark", "--list"])
    assert result.exit_code == 0
