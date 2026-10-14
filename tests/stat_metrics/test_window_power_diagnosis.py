"""Controls for the opt-in diagnosis, independent of fitted model quality."""

import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.shared_types import Batch
from scripts.window_power_diagnosis import feature_gram, summary_features


def batch(times, mask=None):
    t = torch.tensor(times)
    mask = torch.ones_like(t, dtype=torch.bool) if mask is None else torch.tensor(mask)
    return Batch(
        t,
        torch.diff(t, prepend=torch.zeros(len(t), 1)),
        torch.zeros_like(t, dtype=torch.long),
        mask,
    )


def test_coarse_grid_erases_within_bin_timing_and_fine_grid_recovers_it():
    original = batch([[0.2], [2.0]])
    accelerated = batch([[0.1], [2.0]])
    coarse = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)
    fine = SIGKernel(RBFKernel(), "counting_grid", 64, 0, 2)
    a, b = coarse._prepare_kernel(original, accelerated)
    assert torch.equal(a[0], b[0])
    a, b = fine._prepare_kernel(original, accelerated)
    assert not torch.equal(a[0], b[0])


def test_count_and_first_time_controls_separate_timing_from_counts():
    pool = batch([[0.2], [0.1], [0.0]], [[True], [True], [False]])
    before = pool.time_seqs.clone()
    assert feature_gram(pool, 2.0, "count")[0, 1] == 1
    assert feature_gram(pool, 2.0, "first_time")[0, 1] < 1
    assert summary_features(pool, 2.0, "first_time")[2, 0] == 1
    assert torch.equal(before, pool.time_seqs)
    assert pool.no_next_event_mask is None
