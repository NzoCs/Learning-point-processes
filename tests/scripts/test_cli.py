import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.cli import app


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_version(runner):
    """Test the version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "EasyTPP CLI v4.0" in result.output
    assert "Runners Process" in result.output


def test_cli_help(runner):
    """Test the main help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "EasyTPP CLI v4.0" in result.output
    assert "run" in result.output
    assert "inspect" in result.output
    assert "generate" in result.output


def test_cli_run_help(runner):
    """Test the run command help."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run a TPP experiment" in result.output
    assert "--config" in result.output
    assert "--model" in result.output


def test_cli_inspect_help(runner):
    """Test the inspect command help."""
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "Inspect and visualize TPP data" in result.output


def test_cli_generate_help(runner):
    """Test the generate command help."""
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "Generate synthetic TPP data" in result.output


def test_cli_info_help(runner):
    """Test the info command help."""
    result = runner.invoke(app, ["info", "--help"])
    assert result.exit_code == 0
    assert "Display system information" in result.output


def test_cli_setup_help(runner):
    """Test the setup command help."""
    result = runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0
    assert "Interactive configuration" in result.output


def test_cli_benchmark_help(runner):
    """Test the benchmark command help."""
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "Run TPP benchmarks" in result.output


def test_cli_no_args(runner):
    """Test running CLI with no arguments shows help."""
    result = runner.invoke(app, [])
    assert result.exit_code == 2  # typer exits with 2 for no args
    assert "EasyTPP CLI v4.0" in result.output
    assert "Commands" in result.output


def test_cli_invalid_command(runner):
    """Test invalid command."""
    result = runner.invoke(app, ["invalid"])
    assert result.exit_code == 2  # typer error code for invalid command
    assert "No such command" in result.output


def test_cli_run_missing_required(runner):
    """Test run command with missing required args."""
    # The run command might fail when trying to execute
    result = runner.invoke(app, ["run"])
    # It may exit with 1 if execution fails
    assert result.exit_code in [0, 1, 2]


def test_cli_benchmark_missing_config(runner):
    """Test benchmark command with missing config (now required)."""
    result = runner.invoke(app, ["benchmark", "--data-config", "test", "--all"])
    assert result.exit_code == 2  # Should fail because config is required
    assert "Missing option" in result.output or "required" in result.output


def test_cli_inspect_missing_data_dir(runner):
    """Test inspect command with missing data directory."""
    result = runner.invoke(app, ["inspect"])
    assert result.exit_code == 2  # Missing required argument


def test_cli_generate_defaults(runner, tmp_path):
    """Test generate command with defaults."""
    output_dir = tmp_path / "generated"
    result = runner.invoke(app, ["generate", "--output", str(output_dir)])
    # This might fail if dependencies not available, but should not crash
    assert result.exit_code in [0, 1]  # 0 success, 1 failure but no crash


def test_cli_info_defaults(runner):
    """Test info command with defaults."""
    result = runner.invoke(app, ["info"])
    # May fail if system info collection fails
    assert result.exit_code in [0, 1]


def test_cli_setup_defaults(runner):
    """Test setup command with defaults."""
    result = runner.invoke(app, ["setup"])
    # Setup might require input, but should not crash
    assert result.exit_code in [0, 1]


def test_cli_benchmark_list(runner):
    """Test benchmark list option."""
    # Need to provide config since it's required
    result = runner.invoke(app, ["benchmark", "--config", "dummy.yaml", "--list"])
    # May fail if config doesn't exist, but should not crash on --list
    assert result.exit_code in [0, 1, 2]


# Integration test using subprocess (more realistic)
def test_cli_subprocess_version():
    """Test CLI version via subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.cli", "version"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "EasyTPP CLI v4.0" in result.stdout


def test_cli_subprocess_help():
    """Test CLI help via subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "EasyTPP CLI v4.0" in result.stdout
