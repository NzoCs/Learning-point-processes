"""Controls for the opt-in diagnosis, independent of fitted model quality."""

import numpy as np
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test import (
    paired_joint_statistic,
)
from new_ltpp.shared_types import Batch
from scripts.window_first_event_power import (
    first_times,
    p_value,
    pooled_first_time_statistics,
    signed_statistics,
)
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


def test_vectorized_swaps_match_all_eager_statistics():
    rng = np.random.default_rng(92)
    features = rng.normal(size=(8, 3))
    gram = np.exp(-((features[:, None] - features[None, :]) ** 2).sum(-1))
    bits = np.array([[int(i) for i in f"{value:04b}"] for value in range(16)])
    actual = signed_statistics(gram, 1 - 2 * bits)
    eager = []
    for swap in bits:
        x = torch.tensor(np.arange(4) + 4 * swap)
        y = torch.tensor(np.arange(4) + 4 * (1 - swap))
        eager.append(float(paired_joint_statistic(torch.tensor(gram), x, y)))
    np.testing.assert_allclose(actual, eager, rtol=1e-12, atol=1e-14)


def test_first_event_oracle_matches_exact_censored_poisson_delays():
    contexts = np.arange(256) % 2
    rates = np.array([0.2, 2.0])
    targets = np.random.default_rng(31).exponential(size=len(contexts))
    expected = np.minimum(targets / (2 * rates[contexts]), 2.0)
    actual = first_times(
        [lambda t: 0.2 * t, lambda t: 2.0 * t],
        contexts,
        np.random.default_rng(31),
        2.0,
        2.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_pooled_first_time_uses_the_same_kernel_and_permutation_statistic():
    x, y = np.array([0.1, 2.0, 0.7, 2.0]), np.array([0.4, 0.8, 2.0, 2.0])
    weights = np.exp(
        -0.5 * (np.arange(4)[:, None] % 2 - np.arange(4)[None, :] % 2) ** 2
    )
    signs = 1 - 2 * np.random.default_rng(7).integers(2, size=(100, 4))
    pool = np.concatenate((x, y)) / 2.0
    gram = np.exp(-0.5 * (pool[:, None] - pool[None, :]) ** 2 / 0.1**2) * np.tile(
        weights, (2, 2)
    )
    actual = pooled_first_time_statistics(x, y, weights, signs, 2.0)
    np.testing.assert_allclose(
        actual, signed_statistics(gram, signs), rtol=1e-12, atol=1e-14
    )
    assert p_value(np.array([1.0, 1.0 - 1e-13])) == 1.0
