"""Independent finite-sample oracles for the calibration experiment."""

import itertools

import numpy as np
import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    MMDTwoSampleTest,
)
from scripts.scientific_calibration import (
    fast_permutation,
    gram_statistic,
    sample_process,
)


def test_unbiased_statistic_matches_hand_calculation_unequal_groups():
    gram = torch.tensor(
        [
            [5.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 6.0, 7.0, 8.0, 9.0],
            [2.0, 7.0, 10.0, 11.0, 12.0],
            [3.0, 8.0, 11.0, 13.0, 14.0],
            [4.0, 9.0, 12.0, 14.0, 15.0],
        ]
    )
    assert gram_statistic(
        gram, torch.tensor([0, 1]), torch.tensor([2, 3, 4])
    ).item() == pytest.approx(1 + 37 / 3 - 11)


def test_exact_permutation_ranks_control_false_positives_including_ties():
    # Enumerate every possible labelling under exchangeability, no MC uncertainty.
    points = torch.tensor([0.0, 0.0, 1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    gram = torch.exp(-((points[:, None] - points[None, :]) ** 2))
    values = []
    for group in itertools.combinations(range(6), 3):
        other = [i for i in range(6) if i not in group]
        values.append(gram_statistic(gram, torch.tensor(group), torch.tensor(other)))
    stats = torch.stack(values)
    p_values = (stats[:, None] >= stats[None, :]).double().mean(dim=0)
    for alpha in (0.05, 0.1, 0.2, 0.5):
        assert (p_values <= alpha).double().mean() <= alpha + 1e-12


@pytest.mark.parametrize("paired", [False, True])
def test_fast_calibration_matches_real_pysiglib_permutations(paired):
    rng = np.random.default_rng(13)
    x, y = sample_process(rng, "poisson", size=3), sample_process(
        rng, "poisson", size=4 if not paired else 3
    )
    test = MMDTwoSampleTest(SIGKernel(RBFKernel(), "counting_grid", 8, 0, 1), 19)
    torch.manual_seed(41)
    result = test.compute_statistics(x, y, accumulate=False, paired_truncation=paired)
    if paired:
        x, y = test._truncate_to_min_time(x, y)
    torch.manual_seed(41)
    assert fast_permutation(test, x, y) == pytest.approx(result["p_value"].item())
    pooled = test._concat_batches(x, y)
    gram = test.kernel.compute_gram_matrix(pooled, pooled)
    ids = torch.arange(len(pooled.time_seqs))
    expected = gram_statistic(gram, ids[: len(x.time_seqs)], ids[len(x.time_seqs) :])
    torch.testing.assert_close(result["observed_statistic"], expected)


def test_paired_preparation_depends_on_original_labels():
    rng = np.random.default_rng(5)
    x, y = sample_process(rng, "poisson", size=3), sample_process(
        rng, "poisson", size=3, rate=8.0
    )
    test = MMDTwoSampleTest(SIGKernel(RBFKernel(), "counting_grid", 8, 0, 1), 3)
    tx, ty = test._truncate_to_min_time(x, y)
    original = test._concat_batches(tx, ty)
    pool = test._concat_batches(x, y)
    order = torch.tensor([0, 3, 2, 1, 4, 5])
    px, py = test._truncate_to_min_time(
        test._select_batch(pool, order[:3]), test._select_batch(pool, order[3:])
    )
    recomputed = test._concat_batches(px, py)
    assert not torch.equal(
        original.valid_event_mask[order], recomputed.valid_event_mask
    )
