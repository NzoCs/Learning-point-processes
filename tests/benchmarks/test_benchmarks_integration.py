import os
import tempfile
import pytest
from easy_tpp.configs.data_config import DataConfig
from easy_tpp.evaluation.benchmarks.mean_bench import MeanInterTimeBenchmark
from easy_tpp.evaluation.benchmarks.sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)
from easy_tpp.evaluation.benchmarks.sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)
from easy_tpp.evaluation.benchmarks.last_mark_bench import LastMarkBenchmark


# Télécharge un vrai dataset de test depuis HuggingFace
@pytest.fixture(scope="session")
def real_data_config():
    # Utilise la config du dataset test dans runner_config.yaml
    data_config_dict = {
        "train_dir": "NzoCs/test_dataset",
        "valid_dir": "NzoCs/test_dataset",
        "test_dir": "NzoCs/test_dataset",
        "data_format": "json",
        "dataset_id": "NzoCs_test_dataset",
        "data_loading_specs": {"batch_size": 2, "num_workers": 1, "tensor_type": "pt"},
        "data_specs": {"num_event_types": 2, "pad_token_id": 2, "padding_side": "left"},
    }
    return DataConfig.from_dict(data_config_dict)


def _check_json_file_created(save_dir, dataset_name, bench):
    # Use the actual benchmark_name property
    if hasattr(bench, "benchmark_name"):
        bench_name = bench.benchmark_name
    else:
        bench_name = bench
    path = os.path.join(save_dir, dataset_name, f"{bench_name}_results.json")
    assert os.path.exists(path), f"Le fichier {path} n'a pas été créé."
    return path


def test_mean_benchmark_integration(real_data_config, tmp_path):
    bench = MeanInterTimeBenchmark(
        real_data_config, "NzoCs_test_dataset", save_dir=str(tmp_path)
    )
    bench.evaluate()
    _check_json_file_created(str(tmp_path), "NzoCs_test_dataset", bench)


def test_mark_distribution_benchmark_integration(real_data_config, tmp_path):
    bench = MarkDistributionBenchmark(
        real_data_config, "NzoCs_test_dataset", save_dir=str(tmp_path)
    )
    bench.evaluate()
    _check_json_file_created(str(tmp_path), "NzoCs_test_dataset", bench)


def test_intertime_distribution_benchmark_integration(real_data_config, tmp_path):
    bench = InterTimeDistributionBenchmark(
        real_data_config, "NzoCs_test_dataset", save_dir=str(tmp_path)
    )
    bench.evaluate()
    _check_json_file_created(str(tmp_path), "NzoCs_test_dataset", bench)


def test_last_mark_benchmark_integration(real_data_config, tmp_path):
    bench = LastMarkBenchmark(
        real_data_config, "NzoCs_test_dataset", save_dir=str(tmp_path)
    )
    bench.evaluate()
    _check_json_file_created(str(tmp_path), "NzoCs_test_dataset", bench)
