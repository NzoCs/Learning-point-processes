"""Terminal provenance is explicit and observation quota is preserved end to end."""

import pytest
import torch

from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    split_history_future,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.stopped_mmd_test import (
    StoppedMMDTwoSampleTest,
    observe_event_budget,
)
from new_ltpp.shared_types import Batch, uses_stopped_event_budget
from scripts.stopped_conditional_calibration import empty_continuations
from tests.stat_metrics.test_stopped_calibration import make_test


def source():
    t = torch.tensor([[0.1, 0.3, 0.0, 0.0, 0.0], [0.1, 0.3, 0.5, 0.8, 1.0]])
    valid = torch.tensor([[True, True, False, False, False], [True] * 5])
    return Batch(
        t,
        torch.diff(t, prepend=torch.zeros(2, 1)).masked_fill(~valid, 0),
        torch.tensor([[0, 1, 2, 2, 2], [1, 0, 1, 0, 1]]),
        valid,
        sequence_ids=["ended", "quota"],
        no_next_event_mask=torch.tensor([True, False]),
    )


def test_split_retains_an_empty_known_future_and_all_source_ids():
    batch = source()
    histories, future = split_history_future(
        batch, 2, 3, allow_stopped=True, pad_token_id=2
    )
    assert histories.valid_event_mask.all() and histories.time_seqs.shape == (2, 2)
    assert future.sequence_ids == batch.sequence_ids
    assert future.valid_event_mask.tolist() == [[False] * 3, [True] * 3]
    assert future.no_next_event_mask.tolist() == [True, False]
    assert future.type_seqs[0].tolist() == [2] * 3
    assert not future.time_seqs[0].any()
    assert not future.time_delta_seqs[0].any()
    with pytest.raises(ValueError, match="Not enough"):
        split_history_future(batch, 2, 3)
    batch.no_next_event_mask.zero_()
    with pytest.raises(ValueError, match="explicit"):
        split_history_future(batch, 2, 3, allow_stopped=True)


def test_ordinary_complete_test_keeps_stopped_paths_in_every_permutation():
    t = torch.tensor([[0.1, 0.2, 0.3]]).repeat(8, 1)
    full = Batch(
        t,
        torch.diff(t, prepend=torch.zeros(8, 1)),
        torch.zeros_like(t, dtype=torch.long),
        torch.ones_like(t, dtype=torch.bool),
    )
    observed = observe_event_budget(full, 3)
    kernel = make_test().kernel
    test = StoppedMMDTwoSampleTest(kernel, 99)
    torch.manual_seed(55)
    result = test.compute_statistics(observed, empty_continuations(8, 3))
    assert result["p_value"] <= 0.05 and result["observed_statistic"] > 0
    full.valid_event_mask[0, 1:] = False
    with pytest.raises(ValueError, match="explicit"):
        observe_event_budget(full, 3)


def test_terminal_flag_is_loaded_strictly_and_moves_with_the_batch():
    from new_ltpp.data.preprocess.dataset import TPPDataset
    from new_ltpp.data.preprocess.sequence_identity import identify_loaded_data

    payload = dict(
        time_seqs=[[0.1, 0.3]], time_delta_seqs=[[0.1, 0.2]], type_seqs=[[0, 1]]
    )
    loaded = identify_loaded_data(dict(payload), {"no_next_event": [True]})
    assert TPPDataset(loaded)[0]["no_next_event"] is True
    with pytest.raises(ValueError, match="boolean"):
        identify_loaded_data(dict(payload), {"no_next_event": ["yes"]})
    batch = source().to_device(torch.device("meta"))
    assert batch.no_next_event_mask.device == batch.time_seqs.device
    assert uses_stopped_event_budget("FullyNN") and uses_stopped_event_budget("RMTPP")
    assert not uses_stopped_event_budget("NHP")


def test_ordinary_predict_step_passes_the_prepared_observation_to_statistics(tmp_path):
    from types import SimpleNamespace

    from new_ltpp.configs import StatisticalTestConfig
    from new_ltpp.evaluation.accumulators.summary_statistics_accumulator import (
        BatchStatisticsCollector,
    )
    from new_ltpp.models.base.training import TrainingStepsMixin

    config = StatisticalTestConfig(
        test_type="mmd",
        point_process_kernel_type="sig_kernel",
        space_kernel_type="rbf",
        num_event_types=2,
        n_samples=99,
        num_discretization_points=8,
    )
    collector = BatchStatisticsCollector(
        2,
        1.0,
        tmp_path / "statistics",
        config,
        complete_stopped=True,
    )
    t = torch.tensor([[0.1, 0.2, 0.3]]).repeat(8, 1)
    batch = Batch(
        t,
        torch.diff(t, prepend=torch.zeros(8, 1)),
        torch.zeros_like(t, dtype=torch.long),
        torch.ones_like(t, dtype=torch.bool),
    )
    simulator = SimpleNamespace(
        _statistics_collector=collector, conditional_evaluation=None
    )
    model = type("FullyNN", (), {})()
    model._simulator = simulator
    model._io_manager = None
    model.simulate = lambda **kwargs: empty_continuations(8, 3)
    TrainingStepsMixin.predict_step(model, batch, 0)
    stats = collector.compute()["statistical_tests"]
    assert stats["num_sequences"] == 8
    assert stats["p_values"][0] <= 0.05
    assert batch.no_next_event_mask is None
