"""Validate the built distribution away from the editable checkout.

Run `uv build --wheel --no-sources` before this test.
"""

from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_wheel_includes_runtime_packages_and_presets(tmp_path):
    wheels = sorted((ROOT / "dist").glob("new_ltpp-*.whl"))
    if not wheels:
        pytest.skip("Build the wheel with uv build --wheel --no-sources first")
    wheel = max(wheels, key=lambda path: path.stat().st_mtime_ns)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for expected in (
            "new_ltpp/configs/runner_config.py",
            "new_ltpp/evaluation/statistical_testing/point_process_kernels/signature_backend.py",
            "new_ltpp/models/implementations/nhp.py",
            "scripts/cli_runners/experiment_runner.py",
            "yaml_configs/__init__.py",
            "yaml_configs/configs.yaml",
        ):
            assert expected in names
        assert "scripts/compare_signature_backends.py" not in names
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")
        requirements = [
            line for line in metadata.splitlines() if line.startswith("Requires-Dist:")
        ]
        assert "Requires-Dist: pysiglib==4.0.0" in requirements
        assert not any("sigkernel" in line.lower() for line in requirements)
        archive.extractall(tmp_path / "installed")
    # -I excludes the checkout and any PYTHONPATH supplied by an editable install.
    code = (
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from new_ltpp.globals import CONFIGS_FILE, OUTPUT_DIR; "
        "assert CONFIGS_FILE.is_file(); "
        "assert CONFIGS_FILE.is_relative_to(Path(sys.argv[1])); "
        "assert OUTPUT_DIR == Path.cwd() / 'artifacts'; "
        "print(CONFIGS_FILE)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path / "installed")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
