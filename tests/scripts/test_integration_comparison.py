"""Ensure the integration gate detects numerical and structural regressions."""

import pytest

from scripts.integration_suite import compare


@pytest.mark.parametrize(
    "actual",
    [
        {"loss": 1.1, "marks": [0, 1]},
        {"loss": 1.0, "marks": [1, 1]},
        {"loss": 1.0, "marks": [0]},
        {"loss": 1.0},
        {"loss": float("nan"), "marks": [0, 1]},
    ],
)
def test_comparison_rejects_changed_calculations_and_artifact_structure(actual):
    with pytest.raises(AssertionError):
        compare(actual, {"loss": 1.0, "marks": [0, 1]})


def test_comparison_accepts_small_floating_point_roundoff():
    compare({"loss": 1.000001, "marks": [0, 1]}, {"loss": 1.0, "marks": [0, 1]})


def test_results_csv_aligns_different_phase_columns(tmp_path):
    import pandas as pd

    from new_ltpp.evaluation.results_aggregator import ResultsAggregator

    target = tmp_path / "results.csv"
    aggregator = ResultsAggregator(target)
    aggregator.add_result("tiny", {}, test_metrics={"loss": 2.0})
    aggregator.add_result(
        "tiny", {}, sim_metrics={"metrics": {"mmd": 0.3}, "batch_count": 2}
    )
    actual = pd.read_csv(target)
    assert len(actual) == 2
    assert actual.loc[0, "test_loss"] == 2.0
    assert actual.loc[1, "sim_mmd"] == 0.3
    assert pd.isna(actual.loc[0, "sim_mmd"])
    assert pd.isna(actual.loc[1, "test_loss"])


def test_self_correcting_last_step_matches_full_intensity_slice(tmp_path):
    import json

    import torch

    from new_ltpp.configs import RunnerConfig
    from new_ltpp.runners.model_runner import Runner
    from scripts.integration_suite import FIXTURES

    raw = json.loads((FIXTURES / "config.json").read_text())
    raw["model_id"] = raw["model_config"]["model_id"] = "SelfCorrecting"
    raw["save_dir"] = str(tmp_path)
    for key in ("train_dir", "valid_dir", "test_dir"):
        raw["data_config"][key] = str(FIXTURES / "fixture.json")
    runner = Runner(RunnerConfig.model_validate(raw), enable_logging=False)
    runner.datamodule.setup(stage="test")
    batch = next(iter(runner.datamodule.test_dataloader()))
    kwargs = dict(
        batch.to_mapping(), sample_dtimes=torch.full((*batch.time_seqs.shape, 3), 0.2)
    )
    full = runner.model.compute_intensities_at_sample_dtimes(**kwargs)
    last = runner.model.compute_intensities_at_sample_dtimes(
        **kwargs, compute_last_step_only=True
    )
    assert last.shape == (4, 1, 3, 2)
    torch.testing.assert_close(last, full[:, -1:])


def test_parquet_sequence_ids_count_sequences_instead_of_events(tmp_path):
    import pyarrow.parquet as pq
    import torch

    from new_ltpp.models.simulation.tpp_io import SimulationIOManager
    from new_ltpp.shared_types import Batch

    times = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.4, 0.8]])
    batch = Batch(
        times,
        torch.zeros_like(times),
        torch.zeros_like(times, dtype=torch.long),
        torch.ones_like(times, dtype=torch.bool),
    )
    io = SimulationIOManager(1)
    io.setup_io(tmp_path)
    io.append_batch(batch)
    io.append_batch(batch)
    io.finalize()
    records = pq.read_table(tmp_path / "sim_batch.parquet").to_pylist()
    assert len(records) == 12
    assert [row["seq_idx"] for row in records] == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
