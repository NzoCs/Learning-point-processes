"""Validate compatibility without resurrecting obsolete scientific controls."""

import pytest
import torch
from pydantic import ValidationError

from new_ltpp.configs import SimulationConfig, StatisticalTestConfig
from new_ltpp.evaluation.statistical_testing.point_process_kernels.utils import (
    _get_embedding,
)


def test_simulation_has_only_active_seed_and_execution_contract():
    config = SimulationConfig(seed=42)
    assert config.model_dump() == {"seed": 42}
    assert set(SimulationConfig.model_json_schema()["properties"]) == {"seed"}
    contract = config.execution_contract()
    assert contract["mode"] == "fixed_event_count"
    assert contract["event_count_source"] == "input_batch_width"
    assert "ignored_legacy_parameters" not in contract


@pytest.mark.parametrize("field", ["time_window", "batch_size", "initial_buffer_size"])
def test_removed_simulation_controls_are_rejected(field):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SimulationConfig.model_validate({"seed": 42, field: 20})


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_simulation_rejects_seeds_that_lightning_cannot_use(seed):
    with pytest.raises(ValidationError):
        SimulationConfig(seed=seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_simulation_accepts_rng_seed_boundaries(seed):
    assert SimulationConfig(seed=seed).seed == seed


def test_counting_grid_retains_hand_calculated_counts():
    times = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    marks = torch.tensor([[0, 1]])
    actual = _get_embedding(5, "counting_grid", 2, times, marks)
    assert actual[0, 1, 2:].tolist() == [1, 0]
    assert actual[0, 3, 2:].tolist() == [1, 1]


@pytest.mark.parametrize("label", ["linear", "constant", "typo"])
def test_removed_signature_labels_are_rejected_at_all_entry_points(label):
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.sig_kernel import (
        SIGKernel,
    )
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
        LinearKernel,
    )

    times = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    marks = torch.tensor([[0, 1]])
    with pytest.raises(ValueError, match="Unsupported signature path"):
        _get_embedding(5, label, 2, times, marks)
    with pytest.raises(ValueError, match="Unsupported signature path"):
        SIGKernel(LinearKernel(), label, 5, 0, 2)
    with pytest.raises(ValidationError):
        StatisticalTestConfig(
            test_type="mmd",
            point_process_kernel_type="sig_kernel",
            space_kernel_type="linear",
            num_event_types=2,
            n_samples=2,
            embedding_type=label,
        )


def test_signature_config_names_the_actual_representation():
    config = StatisticalTestConfig(
        test_type="mmd",
        point_process_kernel_type="sig_kernel",
        space_kernel_type="linear",
        num_event_types=2,
        n_samples=2,
    )
    assert config.embedding_type == "counting_grid"


def test_simulation_presets_only_describe_active_behavior():
    from new_ltpp.configs.config_utils import load_yaml
    from new_ltpp.globals import CONFIGS_FILE

    presets = load_yaml(CONFIGS_FILE)["simulation_configs"]
    assert set(presets) == {"fixed_events", "quick_test", "debug"}
    for preset in presets.values():
        assert preset == {"seed": 42}
        assert SimulationConfig.model_validate(preset).seed == 42


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
