"""Validate compatibility without resurrecting obsolete scientific controls."""

import pytest
import torch
from pydantic import ValidationError

from new_ltpp.configs import SimulationConfig, StatisticalTestConfig
from new_ltpp.evaluation.statistical_testing.point_process_kernels.utils import (
    _get_embedding,
)


def test_minimal_simulation_config_and_legacy_config_share_execution_policy():
    modern = SimulationConfig(seed=42)
    legacy = SimulationConfig(
        seed=42, time_window=20, batch_size=100, initial_buffer_size=300
    )
    modern_policy = modern.execution_contract()
    legacy_policy = legacy.execution_contract()
    assert legacy_policy.pop("ignored_legacy_parameters") == {
        "time_window": 20,
        "batch_size": 100,
        "initial_buffer_size": 300,
    }
    assert modern_policy.pop("ignored_legacy_parameters") == {}
    assert modern_policy == legacy_policy
    assert modern_policy["mode"] == "fixed_event_count"
    assert modern_policy["event_count_source"] == "input_batch_width"
    restored = SimulationConfig.model_validate(legacy.model_dump(mode="json"))
    assert restored == legacy
    for field in ("time_window", "batch_size", "initial_buffer_size"):
        assert (
            SimulationConfig.model_json_schema()["properties"][field]["deprecated"]
            is True
        )


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_simulation_rejects_seeds_that_lightning_cannot_use(seed):
    with pytest.raises(ValidationError):
        SimulationConfig(seed=seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_simulation_accepts_rng_seed_boundaries(seed):
    assert SimulationConfig(seed=seed).seed == seed


def test_new_and_legacy_embedding_labels_have_identical_hand_counting_paths():
    times = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    marks = torch.tensor([[0, 1]])
    expected = _get_embedding(5, "counting_grid", 2, times, marks)
    for label in ("linear", "constant"):
        torch.testing.assert_close(
            _get_embedding(5, label, 2, times, marks), expected, rtol=0, atol=0
        )
    assert expected[0, 1, 2:].tolist() == [1, 0]
    assert expected[0, 3, 2:].tolist() == [1, 1]
    with pytest.raises(ValueError, match="Unsupported signature path"):
        _get_embedding(5, "typo", 2, times, marks)


def test_signature_config_names_the_actual_representation():
    config = StatisticalTestConfig(
        test_type="mmd",
        point_process_kernel_type="sig_kernel",
        space_kernel_type="linear",
        num_event_types=2,
        n_samples=2,
    )
    assert config.embedding_type == "counting_grid"


def test_old_simulation_preset_names_resolve_without_inactive_controls():
    from new_ltpp.configs.config_utils import load_yaml
    from new_ltpp.globals import CONFIGS_FILE

    presets = load_yaml(CONFIGS_FILE)["simulation_configs"]
    for name in (
        "fixed_events",
        "quick_test",
        "debug",
        "tw30_b5000_b16",
        "tw70_b15000_b32",
        "tw100_b50000_b128",
        "tw100_b50000_b64",
        "tw200_b100000_b128",
        "tw300_b150000_b256",
        "tw80_b30000_b64",
    ):
        assert presets[name] == {"seed": 42}
        assert SimulationConfig.model_validate(presets[name]).seed == 42


def test_compatible_old_manifest_retains_provenance_when_contract_is_added(tmp_path):
    import json

    from new_ltpp.configs import RunnerConfig
    from new_ltpp.runners.run_manifest import RunManifest
    from scripts.integration_suite import FIXTURES

    raw = json.loads((FIXTURES / "config.json").read_text())
    raw["save_dir"] = str(tmp_path)
    for field in ("train_dir", "valid_dir", "test_dir"):
        raw["data_config"][field] = str(FIXTURES / "fixture.json")
    config = RunnerConfig.model_validate(raw)
    manifest = RunManifest(config)
    manifest.start("train")
    original = {
        key: manifest.value[key] for key in ("code", "config", "phases", "created_at")
    }
    del manifest.value["simulation_contract"]
    del manifest.value["numerics"]["signature_path_representation"]
    manifest.save()
    restored = RunManifest(config).value
    assert (
        restored["simulation_contract"] == config.simulation_config.execution_contract()
    )
    assert restored["numerics"]["signature_path_representation"] == "counting_grid"
    assert {key: restored[key] for key in original} == original
