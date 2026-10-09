"""Numerical and artifact contracts for reproducible experiments."""

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    EmbeddingKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.kernel_protocol import (
    PointProcessKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.m_kernel import (
    MKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.shared_types import Batch


def test_marked_gram_preserves_sequence_and_event_indices():
    kernel = EmbeddingKernel(num_classes=3, embedding_dim=2).double()
    with torch.no_grad():
        kernel.emb.weight.copy_(
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]], dtype=torch.float64
            )
        )
    x = torch.tensor([[0, 1, 2], [3, 2, 1]])
    y = torch.tensor([[3, 0], [1, 2], [2, 0]])
    gram = kernel.Gram_matrix(x, y)
    for i in range(2):
        for j in range(3):
            for s in range(3):
                for t in range(2):
                    distance = (
                        (kernel.emb.weight[x[i, s]] - kernel.emb.weight[y[j, t]])
                        .square()
                        .sum()
                    )
                    torch.testing.assert_close(
                        gram[i, j, s, t], torch.exp(-distance / 2)
                    )
    torch.testing.assert_close(gram, kernel.Gram_matrix(y, x).permute(1, 0, 3, 2))


class FixedKernel(PointProcessKernel):
    def __init__(self):
        self.values = torch.tensor(
            [
                [5.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 6.0, 7.0, 8.0, 9.0],
                [2.0, 7.0, 10.0, 11.0, 12.0],
                [3.0, 8.0, 11.0, 13.0, 14.0],
                [4.0, 9.0, 12.0, 14.0, 15.0],
            ]
        )

    def compute_gram_matrix(self, x, y):
        return self.values[x.type_seqs[:, 0]][:, y.type_seqs[:, 0]]


def indexed_batch(indices):
    marks = torch.tensor(indices).reshape(-1, 1)
    times = torch.zeros_like(marks, dtype=torch.float32)
    return Batch(times, times, marks, torch.ones_like(marks, dtype=torch.bool))


def test_generic_mmd_uses_off_diagonal_and_separate_sample_sizes():
    kernel = FixedKernel()
    x, y = indexed_batch([0, 1]), indexed_batch([2, 3, 4])
    # XX off-diagonal average=1, YY=(11+12+14)/3=37/3, XY=(2+3+4+7+8+9)/6=5.5.
    expected = torch.tensor(1 + 37 / 3 - 11)
    torch.testing.assert_close(kernel.compute_mmd(x, y), expected)
    torch.testing.assert_close(kernel.compute_mmd(y, x), expected)
    with pytest.raises(ValueError, match="at least two"):
        kernel.compute_mmd(indexed_batch([0]), y)


def test_zero_distance_transform_is_finite():
    kernel = MKernel(RBFKernel(), EmbeddingKernel(1))
    torch.testing.assert_close(
        kernel._apply_transform(torch.zeros(2, 3)), torch.ones(2, 3)
    )


def test_generated_data_repeats_with_seed_and_keeps_metadata(tmp_path):
    from scripts.cli import app

    runner = CliRunner()
    outputs = []
    for index, seed in enumerate([42, 42, 43]):
        target = tmp_path / str(index)
        result = runner.invoke(
            app,
            [
                "generate",
                "--model",
                "hawkes",
                "--num-sim",
                "10",
                "--num-events-per-seq",
                "5",
                "--burn-in",
                "0",
                "--dim",
                "1",
                "--seed",
                str(seed),
                "--output",
                str(target),
            ],
        )
        assert result.exit_code == 0, result.output
        outputs.append((target / "train.json").read_bytes())
        meta = json.loads((target / "metadata.json").read_text())
        assert meta["simulation_info"]["seed"] == seed
    assert outputs[0] == outputs[1]
    assert outputs[0] != outputs[2]


def make_config(root):
    from new_ltpp.configs import RunnerConfig
    from new_ltpp.globals import CONFIGS_FILE

    return RunnerConfig.from_yaml_presets(
        CONFIGS_FILE,
        config_paths={
            "data_config_path": "data_configs.test",
            "data_loading_config_path": "data_loading_configs.quick_test",
            "general_specs_config_path": "general_specs.quick_test",
            "training_config_path": "training_configs.quick_test",
            "simulation_config_path": "simulation_configs.quick_test",
            "thinning_config_path": "thinning_configs.quick_test",
            "statistical_test_config_path": "statistical_test_configs.sig_kernel",
        },
        model_id="NHP",
        save_dir=str(root),
        max_epochs=1,
    )


def test_effective_config_roundtrip_preserves_simulation_and_output_root(tmp_path):
    from new_ltpp.configs import RunnerConfig

    config = make_config(tmp_path / "run")
    output = tmp_path / "effective.yaml"
    config.save_to_yaml_file(output)
    restored = RunnerConfig.from_yaml(output)
    assert restored.get_yaml_config() == config.get_yaml_config()
    assert restored.simulation_config == config.simulation_config
    assert restored.base_dir == config.base_dir
    assert restored.training_config.seed == 42


def test_dataset_revision_is_passed_to_remote_loader(tmp_path, monkeypatch):
    import datasets
    from new_ltpp.data.preprocess.data_loader import TPPDataModule

    config = make_config(tmp_path)
    data_config = config.data_config.model_copy(update={"revision": "a" * 40})
    observed = []

    def load(source, *, split, revision):
        observed.append((split, revision))
        if split == "dev":
            raise ValueError("dev unavailable")
        return {
            "dim_process": [2],
            "time_since_start": [[0.0, 1.0]],
            "time_since_last_event": [[0.0, 1.0]],
            "type_event": [[0, 1]],
        }

    monkeypatch.setattr(datasets, "load_dataset", load)
    TPPDataModule(data_config)._build_input_from_hf("fixture", "dev")
    assert observed == [("dev", "a" * 40), ("validation", "a" * 40)]


def offline_config(tmp_path, root):
    from new_ltpp.configs import RunnerConfig

    data_file = tmp_path / "fixture.json"
    records = []
    for index in range(8):
        deltas = [0.0, 0.2 + 0.01 * index, 0.3, 0.4, 0.5]
        records.append(
            {
                "dim_process": 2,
                "time_since_start": torch.tensor(deltas).cumsum(0).tolist(),
                "time_since_last_event": deltas,
                "type_event": [0, 1, 0, 1, 0],
            }
        )
    data_file.write_text(json.dumps(records), encoding="utf-8")
    raw = make_config(root).get_yaml_config()
    raw["data_config"].update(
        data_format="json",
        train_dir=str(data_file),
        valid_dir=str(data_file),
        test_dir=str(data_file),
    )
    raw["data_config"]["data_loading_specs"].update(
        num_workers=0, batch_size=4, shuffle=True
    )
    raw["training_config"].update(
        deterministic=True, checkpoints_freq=1, val_freq=1, devices=-1
    )
    raw["model_config"]["thinning_config"].update(num_samples_boundary=5)
    return RunnerConfig.model_validate(raw)


def test_two_short_cpu_trainings_repeat_and_select_generated_checkpoint(tmp_path):
    from new_ltpp.runners.model_runner import Runner

    states = []
    initial = []
    torch.set_num_threads(1)
    for index in range(2):
        config = offline_config(tmp_path, tmp_path / f"run-{index}")
        runner = Runner(config, enable_logging=False)
        initial.append(
            {
                key: value.detach().clone()
                for key, value in runner.model.state_dict().items()
            }
        )
        runner.train()
        assert runner.checkpoint_path and Path(runner.checkpoint_path).is_file()
        checkpoint = torch.load(
            runner.checkpoint_path, map_location="cpu", weights_only=False
        )
        states.append(checkpoint["state_dict"])
        runner.test()
        assert (config.base_dir / "test_results" / "test_results.json").is_file()
    for key in initial[0]:
        torch.testing.assert_close(initial[0][key], initial[1][key], rtol=0, atol=0)
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    assert any(not torch.equal(initial[0][key], states[0][key]) for key in initial[0])


def test_validation_restores_all_cpu_random_streams():
    import random
    import numpy as np
    from new_ltpp.runners.rng_callback import ValidationRNGCallback

    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    callback = ValidationRNGCallback()
    callback.on_validation_start(None, None)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.random()
    np.random.rand(7)
    torch.rand(12)
    callback.on_validation_end(None, None)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)


@pytest.mark.parametrize("count", [0, 1, 2, 7, 11])
def test_dataset_split_keeps_all_ids_once(count):
    from new_ltpp.data.generation.io_simulator import split_records

    records = list(range(count))
    splits = split_records(
        records, {"train": 0.6, "dev": 0.2, "test": 0.2, "unused": 0}
    )
    assert [value for part in splits.values() for value in part] == records
    assert splits["unused"] == []


def test_generation_hashes_match_saved_split_files(tmp_path):
    import hashlib
    from new_ltpp.data.generation.io_simulator import IOSimulator

    records = [{"seq_idx": i, "seq_len": 0} for i in range(7)]
    IOSimulator().save_to_json(records, str(tmp_path), metadata={"seed": 42})
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert (
        sum(metadata["split_info"][f"{name}_size"] for name in ["train", "test", "dev"])
        == 7
    )
    for name, digest in metadata["split_sha256"].items():
        assert (
            hashlib.sha256((tmp_path / f"{name}.json").read_bytes()).hexdigest()
            == digest
        )


def test_run_manifest_persists_configuration_data_hash_and_failure(tmp_path):
    from new_ltpp.runners.run_manifest import RunManifest, sha256_file

    config = offline_config(tmp_path, tmp_path / "run")
    manifest = RunManifest(config)
    manifest.start("train")
    manifest.finish("train", error=RuntimeError("fixture failure"))
    value = json.loads(manifest.path.read_text())
    assert value["phases"]["train"]["status"] == "failed"
    assert value["config"]["simulation_config"] == config.simulation_config.model_dump(
        mode="json"
    )
    assert value["data"]["local_files"]["train"]["sha256"] == sha256_file(
        config.data_config.train_dir
    )
    changed = config.model_copy(
        update={
            "simulation_config": config.simulation_config.model_copy(update={"seed": 7})
        }
    )
    with pytest.raises(ValueError, match="different configuration"):
        RunManifest(changed)


def test_new_attempts_have_separate_directories(tmp_path, monkeypatch):
    monkeypatch.delenv("LTPP_RUN_ID", raising=False)
    first, second = make_config(tmp_path), make_config(tmp_path)
    assert first.run_id != second.run_id
    assert first.base_dir != second.base_dir


def test_marked_mmd_is_invariant_to_swapping_and_permuting_unequal_batches():
    def _make_batch(batch_size, seq_len):
        deltas = torch.linspace(0.1, 1.0, batch_size * seq_len).reshape(
            batch_size, seq_len
        )
        types = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len) % 3
        return Batch(
            deltas.cumsum(1), deltas, types, torch.ones_like(types, dtype=torch.bool)
        )

    torch.manual_seed(9)
    kernel = MKernel(RBFKernel(), EmbeddingKernel(3))
    x, y = _make_batch(batch_size=3, seq_len=4), _make_batch(batch_size=2, seq_len=6)
    value = kernel.compute_mmd(x, y)
    swapped = kernel.compute_mmd(y, x)
    reordered = Batch(**{name: field.flip(0) for name, field in x.to_mapping().items()})
    torch.testing.assert_close(value, swapped)
    torch.testing.assert_close(value, kernel.compute_mmd(reordered, y))


def test_actual_model_simulation_repeats_with_seed(tmp_path):
    from new_ltpp.runners.model_runner import Runner
    from new_ltpp.models.simulation.simulator import Simulator

    runner = Runner(offline_config(tmp_path, tmp_path / "model"), enable_logging=False)
    times = torch.tensor([[0.0, 0.2, 0.5], [0.0, 0.3, 0.7]])
    batch = Batch(
        times,
        torch.diff(times, prepend=torch.zeros(2, 1)),
        torch.tensor([[0, 1, 0], [1, 0, 1]]),
        torch.ones(2, 3, dtype=torch.bool),
    )
    results = []
    runner.model.eval()
    with torch.no_grad():
        for _ in range(2):
            torch.manual_seed(77)
            results.append(
                Simulator(runner.model, runner.config.statistical_test_config).simulate(
                    batch, num_events_to_simulate=2
                )
            )
    for name, value in results[0].to_mapping().items():
        torch.testing.assert_close(value, getattr(results[1], name), rtol=0, atol=0)
        assert torch.isfinite(value).all()


def test_epoch_checkpoint_resume_matches_continuous_training(tmp_path):
    from new_ltpp.configs import RunnerConfig
    from new_ltpp.runners.model_runner import Runner

    def config(root, epochs):
        raw = offline_config(tmp_path, root).get_yaml_config()
        raw["training_config"]["max_epochs"] = epochs
        raw["model_config"]["scheduler_config"].update(max_epochs=2, lr_scheduler=False)
        return RunnerConfig.model_validate(raw)

    continuous = Runner(config(tmp_path / "continuous", 2), enable_logging=False)
    continuous.train()
    interrupted = Runner(config(tmp_path / "interrupted", 1), enable_logging=False)
    interrupted.train()
    checkpoint = str(interrupted.dirpath / "last.ckpt")
    resumed = Runner(
        config(tmp_path / "resumed", 2),
        enable_logging=False,
        checkpoint_path=checkpoint,
    )
    resumed.train()
    for key, value in continuous.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[key], rtol=0, atol=0
        )


def test_cli_replays_effective_configuration_with_complete_cpu_pipeline(
    tmp_path, monkeypatch
):
    from scripts.cli import app

    monkeypatch.delenv("LTPP_RUN_ID", raising=False)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    config = offline_config(tmp_path, tmp_path / "original")
    raw = config.get_yaml_config()
    raw["statistical_test_config"].update(n_samples=2, num_discretization_points=8)
    from new_ltpp.configs import RunnerConfig

    config = RunnerConfig.model_validate(raw)
    config_path = tmp_path / "effective.yaml"
    config.save_to_yaml_file(config_path)
    output = tmp_path / "replayed"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--config",
            str(config_path),
            "--phase",
            "all",
            "--save-dir",
            str(output),
            "--seed",
            "123",
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    manifests = list(output.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["config"]["training_config"]["seed"] == 123
    assert all(
        manifest["phases"][phase]["status"] == "completed"
        for phase in ["train", "test", "predict"]
    )
    assert (
        manifest["phases"]["test"]["checkpoint"]["sha256"]
        == manifest["phases"]["train"]["checkpoint"]["sha256"]
    )


def test_generation_preserves_empty_sequences_and_stable_tied_marks():
    import numpy as np
    from new_ltpp.data.generation.simulation_manager import SimulationManager

    manager = SimulationManager(lambda: None, 2, 3, 0)
    records = manager.format_simulations(
        [
            (np.array([], dtype=float), np.array([], dtype=int)),
            (np.array([1.0, 1.0, 2.0]), np.array([1, 0, 1])),
        ]
    )
    assert [record["seq_idx"] for record in records] == [0, 1]
    assert records[0]["seq_len"] == 0
    assert records[0]["time_since_start"] == []
    assert records[1]["type_event"] == [1, 0, 1]


def test_simulation_p_value_uses_effective_draw_count():
    from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
        MMDTwoSampleTest,
    )

    class ConstantKernel(FixedKernel):
        def compute_mmd(self, x, y):
            return torch.tensor(0.0)

    batch = indexed_batch([0, 1])
    test = MMDTwoSampleTest(ConstantKernel(), n_samples=30)
    result = test.compute_statistics(batch, batch, simulations=[batch, batch, batch])
    assert result["num_null_samples"] == 2
    assert result["null_method"] == "simulation_comparison"
    assert result["p_value"].item() == 1.0
    assert test.get_final_p_value().item() == 1.0


def test_device_request_is_explicit_and_cpu_is_selectable(monkeypatch):
    from new_ltpp.runners.trainer_factory import TrainerFactory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert TrainerFactory._resolve_devices(-1) == (1, "cpu")
    with pytest.raises(RuntimeError, match="GPU requested"):
        TrainerFactory._resolve_devices(1)


def test_unsupported_ksd_is_rejected_before_execution():
    from pydantic import ValidationError
    from new_ltpp.configs import StatisticalTestConfig

    with pytest.raises(ValidationError):
        StatisticalTestConfig(
            test_type="ksd",
            point_process_kernel_type="sig_kernel",
            space_kernel_type="linear",
            num_event_types=2,
            n_samples=10,
        )


def test_nhp_simulates_without_fictitious_initial_event(tmp_path):
    from new_ltpp.runners.model_runner import Runner
    from new_ltpp.models.simulation.simulator import Simulator

    runner = Runner(
        offline_config(tmp_path, tmp_path / "unconditional"), enable_logging=False
    )
    runner.model._simulator = Simulator(
        runner.model, runner.config.statistical_test_config
    )
    runner.model.eval()
    with torch.no_grad():
        result = runner.model.simulate_from_scratch(num_sequences=2, num_events=3)
        sample_times = torch.tensor([[[0.0, 0.2]]]).expand(2, 1, 2)
        empty = torch.empty(2, 0)
        intensity = runner.model.compute_intensities_at_sample_dtimes(
            time_delta_seqs=empty,
            type_seqs=empty.long(),
            sample_dtimes=sample_times,
            compute_last_step_only=True,
        )
        expected = runner.model.layer_intensity(
            torch.zeros(2, 1, 2, runner.model.hidden_size)
        )
        torch.testing.assert_close(intensity, expected)
    assert result.time_seqs.shape == (2, 3)
    assert result.valid_event_mask.all()
    assert (result.time_seqs[:, 0] > 0).all()
    assert torch.isfinite(result.time_seqs).all()
