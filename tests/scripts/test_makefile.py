"""Tests for Makefile commands.

Each test invokes a `make <target>` command via subprocess from the project root
and asserts that it exits successfully (or with an expected non-zero code for
targets that require user input / external resources).
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Project root = 2 levels up from this file (tests/scripts/ → tests/ → project/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _make(*targets: str, env: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run `make <target>` in the project root and return the CompletedProcess."""
    cmd = ["make", *targets]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=timeout,
        env=env,
    )


# ---------------------------------------------------------------------------
# Targets that should always succeed
# ---------------------------------------------------------------------------

def test_make_help():
    """`make help` lists all available commands."""
    result = _make("help")
    assert result.returncode == 0, f"make help failed:\n{result.stderr}"
    # The help output should reference at least a few known targets
    assert "install" in result.stdout or "install" in result.stderr


def test_make_cli_help():
    """`make cli-help` runs `new-ltpp --help`."""
    result = _make("cli-help")
    assert result.returncode == 0, f"make cli-help failed:\n{result.stderr}"


def test_make_version():
    """`make version` prints the CLI version."""
    result = _make("version")
    assert result.returncode == 0, f"make version failed:\n{result.stderr}"


def test_make_info():
    """`make info` displays system / environment info."""
    result = _make("info")
    assert result.returncode == 0, f"make info failed:\n{result.stderr}"


def test_make_benchmark_list():
    """`make benchmark-list` lists available benchmarks."""
    result = _make("benchmark-list")
    assert result.returncode == 0, f"make benchmark-list failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Targets that need mandatory variables → expected to fail with a clear error
# ---------------------------------------------------------------------------

def test_make_run_without_args():
    """`make run` without MODEL_ID / DATA should exit non-zero with a hint."""
    result = _make("run")
    assert result.returncode != 0, "make run should fail when MODEL_ID / DATA are missing"
    combined = result.stdout + result.stderr
    assert "MODEL_ID" in combined or "required" in combined.lower(), (
        f"Expected a 'MODEL_ID required' message, got:\n{combined}"
    )


# ---------------------------------------------------------------------------
# Code-quality targets (lint / format / type-check) – allowed to report issues
# ---------------------------------------------------------------------------

def test_make_lint():
    """`make lint` runs flake8 – may report issues but must not crash."""
    result = _make("lint")
    # flake8 exits 0 when no issues, 1 when issues found; both are fine here.
    assert result.returncode in (0, 1), (
        f"make lint exited unexpectedly ({result.returncode}):\n{result.stderr}"
    )


def test_make_type_check():
    """`make type-check` runs mypy – may report issues but must not crash."""
    result = _make("type-check")
    assert result.returncode in (0, 1), (
        f"make type-check exited unexpectedly ({result.returncode}):\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Slow / resource-heavy targets – skipped in CI by default
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_make_run_demo():
    """`make run-demo` runs the full demo pipeline (slow)."""
    result = _make("run-demo", timeout=600)
    assert result.returncode == 0, f"make run-demo failed:\n{result.stderr}"


@pytest.mark.slow
def test_make_all_benchmarks():
    """`make all-benchmarks` runs benchmarks on multiple configs (slow)."""
    result = _make("all-benchmarks", timeout=600)
    assert result.returncode == 0, f"make all-benchmarks failed:\n{result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
