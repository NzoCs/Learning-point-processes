"""Prevent fitting benchmark baselines on held-out evaluation data."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from new_ltpp.configs import DataConfig
from new_ltpp.data.preprocess.data_loader import TPPDataModule
from new_ltpp.data.preprocess.visualizer import Visualizer
from new_ltpp.evaluation.benchmarks.benchmark_manager import BenchmarkManager
from new_ltpp.evaluation.benchmarks.mean_bench import MeanInterTimeBenchmark
from new_ltpp.evaluation.benchmarks.sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)
from new_ltpp.evaluation.benchmarks.sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)


@pytest.fixture
def config(tmp_path):
    raw = json.loads((Path(__file__).parent / "integration/config.json").read_text())[
        "data_config"
    ]
    source = json.loads(
        (Path(__file__).parent / "integration/fixture.json").read_text()
    )
    for split in ("train", "valid", "test"):
        rows = json.loads(json.dumps(source[:2]))
        for row in rows:
            row["time_since_last_event"] = (
                [0.0, 2.0, 2.0, 2.0, 2.0]
                if split != "test"
                else [0.0, 20.0, 20.0, 20.0, 20.0]
            )
            row["time_since_start"] = np.cumsum(row["time_since_last_event"]).tolist()
            row["type_event"] = [0] * 5 if split != "test" else [1] * 5
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(rows))
        raw[f"{split}_dir"] = str(path)
    return DataConfig.model_validate(raw)


def test_mean_baseline_fits_training_split_and_exports_evaluation(config, tmp_path):
    bench = MeanInterTimeBenchmark(config, tmp_path)
    with pytest.raises(ValueError, match="computed"):
        bench._create_dtime_predictions(next(iter(bench.data_module.test_dataloader())))
    result = bench.evaluate()
    assert result["mean_inter_time_used"] == pytest.approx(1.6)
    assert result["num_batches_evaluated"] == 1
    assert result["metrics"] and all(np.isfinite(v) for v in result["metrics"].values())
    bench.evaluate()
    with (tmp_path / "benchmarks_results.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 and float(rows[0]["mean_inter_time_used"]) == pytest.approx(
        1.6
    )


def test_empirical_time_distribution_uses_training_and_samples_its_support(
    config, tmp_path
):
    bench = InterTimeDistributionBenchmark(config, tmp_path, num_bins=4)
    with pytest.raises(ValueError, match="computed"):
        bench._sample_from_distribution((2, 3))
    result = bench.evaluate()
    torch.testing.assert_close(bench.bin_probabilities.sum(), torch.tensor(1.0))
    torch.testing.assert_close(bench.bin_centers, torch.full((4,), 2.0))
    assert result["num_bins"] == 4
    torch.testing.assert_close(
        bench._sample_from_distribution((2, 3)), torch.full((2, 3), 2.0)
    )


def test_empirical_marks_use_training_not_test(config, tmp_path):
    bench = MarkDistributionBenchmark(config, tmp_path)
    with pytest.raises(ValueError, match="initialized"):
        bench._sample_from_distribution((2, 3))
    result = bench.evaluate()
    torch.testing.assert_close(bench.mark_probabilities, torch.tensor([1.0, 0.0]))
    assert (bench._sample_from_distribution((2, 3)) == 0).all()
    assert result["distribution_stats"]["entropy"] == pytest.approx(0.0)


def test_manager_runs_real_benchmarks_and_rejects_unknown_names(config, tmp_path):
    manager = BenchmarkManager(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        manager.run_by_names(["unknown"], config)
    results = manager.run_all_benchmarks(config)
    assert len(results) == 4
    assert all(r["num_batches_evaluated"] == 1 for r in results.values())
    assert manager.run_all_benchmarks(
        [config, config.model_copy(update={"dataset_id": "second"})]
    ).keys() == {"test", "second"}


def test_visualizer_preserves_event_counts_and_exports_figures(config, tmp_path):
    vis = Visualizer(TPPDataModule(config), save_dir=str(tmp_path), max_events=7)
    assert len(vis.all_event_types) == len(vis.all_time_deltas) == 7
    assert vis.seq_lengths.tolist() == [5, 5]
    assert (vis.all_event_types == 1).all()
    vis.plot_event_types()
    vis.plot_inter_event_times()
    vis.plot_sequence_lengths()
    for name in (
        "event_type_dist.png",
        "inter_event_time_dist.png",
        "sequence_length_dist.png",
    ):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="not valid"):
        Visualizer(TPPDataModule(config), split="unknown")
