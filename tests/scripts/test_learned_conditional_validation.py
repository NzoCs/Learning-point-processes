"""Predeclared alternatives and exact comparisons must not hide numerical drift."""

import pytest
import torch

from new_ltpp.shared_types import Batch
from scripts.conditional_reproducibility import exact
from scripts.learned_conditional_calibration import alternative


def test_alternatives_preserve_inputs_history_and_relative_origin():
    times = torch.tensor([[0.2, 0.4], [0.3, 0.6]])
    gaps = torch.tensor([[0.2, 0.2], [0.3, 0.3]])
    original = Batch(
        times, gaps, torch.tensor([[1, 1], [0, 0]]), torch.ones(2, 2, dtype=torch.bool)
    )
    history = Batch(
        times.clone(),
        gaps.clone(),
        torch.tensor([[0, 0], [1, 1]]),
        torch.ones(2, 2, dtype=torch.bool),
    )
    changed = alternative(original, history, "time_scale_x4")
    torch.testing.assert_close(changed.time_seqs, times * 4)
    torch.testing.assert_close(changed.time_delta_seqs, gaps * 4)
    marks = alternative(original, history, "context_marks")
    assert marks.type_seqs.tolist() == [[0, 0], [1, 1]]
    assert original.type_seqs.tolist() == [[1, 1], [0, 0]]
    torch.testing.assert_close(
        original.time_seqs, torch.tensor([[0.2, 0.4], [0.3, 0.6]])
    )
    torch.testing.assert_close(marks.time_seqs, original.time_seqs)
    torch.testing.assert_close(changed.valid_event_mask, original.valid_event_mask)
    with pytest.raises(ValueError):
        alternative(original, history, "unknown")


def test_repeat_comparison_has_zero_numerical_tolerance():
    exact(
        {"state": [torch.tensor([1.0]), 0.05]}, {"state": [torch.tensor([1.0]), 0.05]}
    )
    with pytest.raises(AssertionError):
        exact(torch.tensor([1.0]), torch.tensor([1.000001]))
    with pytest.raises(AssertionError):
        exact([0.05], [0.05000000000001])
    with pytest.raises(AssertionError):
        exact({"state": 1}, {"other": 1})


def test_nhp_conditional_epoch_resume_with_active_lr_scheduler(tmp_path):
    import json

    import pyarrow.parquet as pq

    from new_ltpp.configs import RunnerConfig
    from new_ltpp.runners.model_runner import Runner
    from scripts.conditional_integration_suite import make_splits
    from scripts.integration_suite import FIXTURES

    torch.set_num_threads(1)
    sources = {}
    for split, rows in make_splits().items():
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(rows))
        sources[split] = str(path)

    def configuration(name, epochs):
        raw = json.loads((FIXTURES / "config.json").read_text())
        raw["save_dir"] = str(tmp_path / name)
        raw["training_config"].update(max_epochs=epochs, lr_scheduler=True)
        raw["model_config"]["scheduler_config"].update(max_epochs=2, lr_scheduler=True)
        raw["simulation_config"]["conditional_evaluation"] = {
            "history_events": 2,
            "future_events": 3,
        }
        raw["statistical_test_config"].update(n_samples=3, num_discretization_points=8)
        raw["data_config"].update(
            train_dir=sources["train"],
            valid_dir=sources["dev"],
            test_dir=sources["test"],
        )
        return RunnerConfig.model_validate(raw)

    continuous = Runner(configuration("continuous", 2), enable_logging=False)
    continuous.train()
    interrupted = Runner(configuration("interrupted", 1), enable_logging=False)
    interrupted.train()
    resumed = Runner(
        configuration("resumed", 2),
        enable_logging=False,
        checkpoint_path=str(interrupted.dirpath / "last.ckpt"),
    )
    resumed.train()
    a = torch.load(
        continuous.dirpath / "last.ckpt", map_location="cpu", weights_only=False
    )
    b = torch.load(
        resumed.dirpath / "last.ckpt", map_location="cpu", weights_only=False
    )
    assert len(a["lr_schedulers"]) == 1
    assert a["lr_schedulers"][0]["last_epoch"] == 2
    assert a["optimizer_states"][0]["param_groups"][0]["lr"] == 0
    for key in (
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "epoch",
        "global_step",
    ):
        exact(a[key], b[key])
    continuous.predict()
    resumed.predict()
    exact(
        continuous.model._simulator._statistics_collector.compute()[
            "statistical_tests"
        ],
        resumed.model._simulator._statistics_collector.compute()["statistical_tests"],
    )

    def events(runner):
        return pq.read_table(
            next(runner.config.base_dir.rglob("*.parquet"))
        ).to_pylist()

    exact(events(continuous), events(resumed))
