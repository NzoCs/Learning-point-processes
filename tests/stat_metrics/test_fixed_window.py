"""Fixed windows retain empty censored futures without declaring permanent ends."""

import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
)
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import heterogeneous_histories


def test_window_keeps_empty_futures_and_last_event_differences():
    n = 16
    h = heterogeneous_histories(n)
    t = torch.tensor([[0.1, 0.3]]).repeat(n, 1)
    full = Batch(
        t,
        torch.diff(t, prepend=torch.zeros(n, 1)),
        torch.zeros_like(t, dtype=torch.long),
        torch.ones_like(t, dtype=torch.bool),
    )
    empty = Batch(
        torch.zeros_like(t),
        torch.zeros_like(t),
        torch.full_like(full.type_seqs, 2),
        torch.zeros_like(t, dtype=torch.bool),
    )
    test = ConditionalMMDTwoSampleTest(
        SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2), 99, observation_horizon=1.0
    )
    torch.manual_seed(42)
    result = test.compute_statistics(empty, full, histories=h)
    assert result["p_value"] <= 0.05 and result["observed_statistic"] > 0
    assert empty.no_next_event_mask is None
    full.time_seqs[0, -1] = 1.1
    with pytest.raises(ValueError, match="fixed horizon"):
        test.compute_statistics(empty, full, histories=h)
    full.time_seqs[0, -1] = 0.3
    empty.no_next_event_mask = torch.ones(n, dtype=torch.bool)
    with pytest.raises(ValueError, match="permanent termination"):
        test.compute_statistics(empty, full, histories=h)


def test_window_sampler_matches_known_marked_poisson_and_safety_budget(tmp_path):
    import math

    import numpy as np

    from scripts.fixed_window_calibration import window_continuations
    from scripts.independent_simulator_validation import constant_poisson_model, history

    torch.set_num_threads(1)
    model, rates = constant_poisson_model("RMTPP", tmp_path)
    h = history(1024)
    torch.manual_seed(615)
    result = window_continuations(model, h, 0.5)
    counts = result.valid_event_mask.sum(-1).double()
    expected = sum(rates) * 0.5
    assert abs(counts.mean().item() - expected) < 0.12
    assert abs((counts == 0).double().mean().item() - math.exp(-expected)) < 0.05
    assert result.no_next_event_mask is None
    oracle = window_continuations(model, h, 0.5, rng=np.random.default_rng(7615))
    assert abs(oracle.valid_event_mask.sum(-1).double().mean().item() - expected) < 0.12
    with pytest.raises(RuntimeError, match="budget exhausted"):
        window_continuations(model, h, 0.5, max_events=1)
