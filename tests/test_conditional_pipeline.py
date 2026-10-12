"""Holdout prediction through the production loader, model and statistical collector."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from new_ltpp.configs import RunnerConfig, SimulationConfig, StatisticalTestConfig
from new_ltpp.data.preprocess.dataset import TPPDataset
from new_ltpp.data.preprocess.sequence_identity import identify_loaded_data
from new_ltpp.evaluation.accumulators.summary_statistics_accumulator import (
    BatchStatisticsCollector,
)
from new_ltpp.models.base.training import TrainingMixin
from new_ltpp.models.simulation.tpp_io import SimulationIOManager
from new_ltpp.shared_types import Batch
from tests.test_reproducibility import offline_config


def identified_batch():
    times = torch.tensor([[0.0, 1.0, 2.0, 4.0, 7.0], [0.0, 2.0, 3.0, 5.0, 8.0]])
    return Batch(
        times,
        torch.diff(times, prepend=torch.zeros(2, 1)),
        torch.tensor([[0, 1, 0, 1, 0]] * 2),
        torch.ones(2, 5, dtype=torch.bool),
        sequence_ids=["sequence-a", "sequence-b"],
    )


def test_conditional_configuration_roundtrip_and_legacy_serialization():
    assert SimulationConfig(seed=42).model_dump() == {"seed": 42}
    raw = {
        "seed": 42,
        "conditional_evaluation": {"history_events": 2, "future_events": 3},
    }
    config = SimulationConfig.model_validate(raw)
    assert json.loads(config.model_dump_json()) == raw
    assert config.execution_contract()["mode"] == "conditional_holdout"
    with pytest.raises(ValueError):
        SimulationConfig(
            conditional_evaluation={"history_events": 0, "future_events": 3}
        )


def test_identity_is_stable_and_explicit_ids_survive_dataset_selection():
    data = {
        "time_seqs": [[0.0, 1.0], [0.0, 2.0]],
        "time_delta_seqs": [[0.0, 1.0], [0.0, 2.0]],
        "type_seqs": [[0, 1], [1, 0]],
    }
    first = TPPDataset(data)
    second = TPPDataset(json.loads(json.dumps(data)))
    assert first.sequence_ids == second.sequence_ids
    assert first[1]["sequence_id"] == second[1]["sequence_id"]
    explicit = identify_loaded_data(
        dict(data), {"sequence_id": ["subject-a", "subject-b"]}
    )
    dataset = TPPDataset(explicit)
    assert [dataset[i]["sequence_id"] for i in [1, 0]] == [
        'provided:"subject-b"',
        'provided:"subject-a"',
    ]


def fake_model(tmp_path):
    config = StatisticalTestConfig(
        test_type="mmd",
        point_process_kernel_type="sig_kernel",
        space_kernel_type="rbf",
        num_event_types=2,
        n_samples=3,
        num_discretization_points=8,
    )
    collector = BatchStatisticsCollector(2, 10.0, tmp_path, config, conditional=True)
    conditional = SimulationConfig(
        conditional_evaluation={"history_events": 2, "future_events": 3}
    )
    model = SimpleNamespace(
        _simulator=SimpleNamespace(
            conditional_evaluation=conditional.conditional_evaluation,
            _statistics_collector=collector,
            last_result=None,
        ),
        _io_manager=SimulationIOManager(2, preserve_origin=True),
        calls=[],
    )

    def simulate(history, num_events_to_simulate):
        model.calls.append(history)
        assert num_events_to_simulate == 3
        times = torch.tensor([[0.5, 1.0, 2.0], [0.6, 1.2, 2.4]])
        return Batch(
            times,
            torch.diff(times, prepend=torch.zeros(2, 1)),
            torch.tensor([[0, 1, 0]] * 2),
            torch.ones(2, 3, dtype=torch.bool),
        )

    model.simulate = simulate
    return model


def test_predict_holds_out_future_and_rejects_reuse_before_simulation(tmp_path):
    model = fake_model(tmp_path)
    batch = identified_batch()
    original = batch.time_seqs.clone()
    collector = model._simulator._statistics_collector
    captured = []
    original_update = collector.update

    def record_update(future, simulation, **kwargs):
        captured.append(future)
        return original_update(future, simulation, **kwargs)

    collector.update = record_update
    TrainingMixin.predict_step(model, batch, 0)
    torch.testing.assert_close(model.calls[0].time_seqs, original[:, :2])
    torch.testing.assert_close(
        captured[0].time_seqs, original[:, 2:] - original[:, 1:2]
    )
    torch.testing.assert_close(batch.time_seqs, original)
    assert model.calls[0].sequence_ids == batch.sequence_ids
    assert model._simulator.last_result.sequence_ids == batch.sequence_ids
    stats = collector.compute()["statistical_tests"]
    assert stats["num_sequences"] == 2
    assert stats["null_methods"] == ["within_history_pair_swaps_joint_kernel_v1"]
    rows = model._io_manager._accumulated_data
    assert rows[0]["sequence_id"] == "sequence-a"
    assert rows[0]["time_since_start"] == rows[0]["time_since_last_event"] == 0.5
    with pytest.raises(ValueError):
        TrainingMixin.predict_step(model, batch, 1)
    assert len(model.calls) == 1
    assert collector._batch_count == 1


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "short", "singleton"])
def test_invalid_conditional_inputs_fail_before_generation(tmp_path, invalid):
    model = fake_model(tmp_path)
    batch = identified_batch()
    if invalid == "missing":
        batch.sequence_ids = None
    elif invalid == "duplicate":
        batch.sequence_ids = ["same", "same"]
    elif invalid == "short":
        batch.valid_event_mask[:, -1] = False
    else:
        batch = Batch(
            *(value[:1] for value in batch.to_mapping().values()), sequence_ids=["only"]
        )
    with pytest.raises(ValueError):
        TrainingMixin.predict_step(model, batch, 0)
    assert model.calls == []
    assert model._simulator._statistics_collector._batch_count == 0


def test_conditional_ddp_is_rejected_before_setup(tmp_path):
    from new_ltpp.runners.callbacks import PredictionStatsCallback

    callback = PredictionStatsCallback.__new__(PredictionStatsCallback)
    callback.simulation_config = SimulationConfig(
        conditional_evaluation={"history_events": 2, "future_events": 3}
    )
    with pytest.raises(ValueError, match="single process"):
        callback.on_predict_start(SimpleNamespace(world_size=2), None)


def test_conditional_collector_requires_test_configuration(tmp_path):
    with pytest.raises(ValueError, match="statistical test configuration"):
        BatchStatisticsCollector(2, 10.0, tmp_path, conditional=True)


@pytest.mark.parametrize("model_id", ["NHP", "Hawkes"])
def test_complete_cli_conditional_pipeline(tmp_path, monkeypatch, model_id):
    from scripts.cli import app

    monkeypatch.delenv("LTPP_RUN_ID", raising=False)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    collected = []
    compute = BatchStatisticsCollector.compute

    def record_compute(collector):
        stats = compute(collector)
        collected.append(
            (
                stats["statistical_tests"],
                set(collector._accumulators["statistical_tests"].test._seen_pair_ids),
            )
        )
        return stats

    monkeypatch.setattr(BatchStatisticsCollector, "compute", record_compute)
    torch.set_num_threads(1)
    raw = offline_config(tmp_path, tmp_path / "configured").get_yaml_config()
    data_path = tmp_path / "fixture.json"
    records = json.loads(data_path.read_text())
    for index, record in enumerate(records):
        record["sequence_id"] = f"independent-sequence-{index}"
    data_path.write_text(json.dumps(records), encoding="utf-8")
    raw["model_id"] = raw["model_config"]["model_id"] = model_id
    raw["simulation_config"]["conditional_evaluation"] = {
        "history_events": 2,
        "future_events": 3,
    }
    raw["statistical_test_config"].update(n_samples=3, num_discretization_points=8)
    config = RunnerConfig.model_validate(raw)
    path = tmp_path / "conditional.yaml"
    config.save_to_yaml_file(path)
    output = tmp_path / "output"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--config",
            str(path),
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
    assert all(
        manifest["phases"][phase]["status"] == "completed"
        for phase in ["train", "test", "predict"]
    )
    prediction = manifest["phases"]["predict"]
    assert (
        prediction["statistical_protocol"]["null_method"]
        == "within_history_pair_swaps_joint_kernel_v1"
    )
    assert (
        prediction["statistical_interpretation"]["fit_independence"]
        == "not_verified_by_pipeline"
    )
    files = list(output.rglob("*.parquet"))
    assert len(files) == 1
    events = pd.read_parquet(files[0])
    assert len(events) == 24
    assert events.groupby("sequence_id").size().tolist() == [3] * 8
    assert set(events["sequence_id"]) == {
        f'provided:"independent-sequence-{i}"' for i in range(8)
    }
    stats, seen_ids = collected[-1]
    assert seen_ids == set(events["sequence_id"])
    assert stats["num_sequences"] == 8
    assert stats["null_methods"] == ["within_history_pair_swaps_joint_kernel_v1"] * 2
    from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
        monte_carlo_p_value,
    )

    observed = torch.tensor(stats["observed_statistic"], dtype=torch.float64).sum()
    null = (
        torch.tensor(stats["permuted_statistic"], dtype=torch.float64)
        .reshape(2, 3)
        .sum(0)
    )
    assert stats["pooled_p_value"] == pytest.approx(
        monte_carlo_p_value(observed, null).item()
    )
    first = events[events["event_idx"] == 0]
    assert (first["time_since_last_event"] > 0).all()
    assert first["time_since_start"].tolist() == pytest.approx(
        first["time_since_last_event"].tolist(), abs=1e-6
    )
