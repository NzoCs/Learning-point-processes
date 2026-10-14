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
