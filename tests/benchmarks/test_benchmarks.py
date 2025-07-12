import os
import tempfile
import yaml
import torch
import numpy as np
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.evaluation.benchmarks.mean_bench import MeanInterTimeBenchmark
from easy_tpp.evaluation.benchmarks.sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)
from easy_tpp.evaluation.benchmarks.sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)
from easy_tpp.evaluation.benchmarks.last_mark_bench import LastMarkBenchmark
from unittest.mock import patch

# Minimal DataConfig for testing (adapt as needed for your project)
MINIMAL_DATA_CONFIG = {
    "train_dir": "dummy_train.csv",
    "valid_dir": "dummy_valid.csv",
    "test_dir": "dummy_test.csv",
    "data_format": "csv",
    "dataset_id": "dummy",
    "data_loading_specs": {"batch_size": 2, "num_workers": 0, "tensor_type": "pt"},
    "data_specs": {"num_event_types": 2},
}

EXPERIMENT_ID = "dummy_exp"


def get_dummy_data_config():
    return DataConfig.from_dict(MINIMAL_DATA_CONFIG)


def patch_setup():
    """Patch TPPDataModule.setup to do nothing during tests."""
    return patch(
        "easy_tpp.preprocess.data_loader.TPPDataModule.setup",
        lambda self, mode=None: None,
    )


def test_mean_inter_time_benchmark_runs():
    data_config = get_dummy_data_config()
    with patch_setup():
        bench = MeanInterTimeBenchmark(data_config, EXPERIMENT_ID)
        bench.data_module.train_dataloader = lambda: []
        bench.data_module.test_dataloader = lambda: []
        bench.mean_inter_time = 1.0  # Fake value for test
        result = bench._prepare_results({"time_rmse_mean": 0.0}, 0)
        assert "mean_inter_time_used" in result


def test_mark_distribution_benchmark_runs():
    data_config = get_dummy_data_config()
    with patch_setup():
        bench = MarkDistributionBenchmark(data_config, EXPERIMENT_ID)
        bench.data_module.train_dataloader = lambda: []
        bench.data_module.test_dataloader = lambda: []
        bench.mark_probabilities = np.array([0.5, 0.5])
        result = bench._prepare_results({"type_accuracy_mean": 1.0}, 0)
        assert "distribution_stats" in result


def test_intertime_distribution_benchmark_runs():
    data_config = get_dummy_data_config()
    with patch_setup():
        bench = InterTimeDistributionBenchmark(data_config, EXPERIMENT_ID)
        bench.data_module.train_dataloader = lambda: []
        bench.data_module.test_dataloader = lambda: []
        bench.bin_probabilities = np.array([1.0])
        bench.bin_centers = np.array([1.0])
        # Mock required abstract methods if needed
        if not hasattr(bench, "benchmark_name"):
            bench.benchmark_name = "intertime_distribution"
        if not hasattr(bench, "_prepare_benchmark"):
            bench._prepare_benchmark = lambda: None
        result = bench._prepare_results({"time_rmse_mean": 0.0}, 0)
        assert "distribution_stats" in result


def test_last_mark_benchmark_runs():
    data_config = get_dummy_data_config()
    with patch_setup():
        bench = LastMarkBenchmark(data_config, EXPERIMENT_ID)
        bench.data_module.train_dataloader = lambda: []
        bench.data_module.test_dataloader = lambda: []
        bench._create_predictions = lambda: 0
        result = bench._prepare_results({"type_accuracy_mean": 1.0}, 0)
        assert "strategy" in result
