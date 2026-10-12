"""Check the experimental protocol before interpreting empirical calibration."""

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
from new_ltpp.shared_types import Batch
from scripts.compare_truncation import (
    RecomputedTruncationTest,
    count_ge,
    paired_statistics,
    permutation_p_value,
)
from scripts.scientific_calibration import sample_process


def truncate_oracle(x, y):
    # Scalar implementation with explicit event lists, independent of tensor helper.
    masks = [b.valid_event_mask.clone() for b in (x, y)]
    for i in range(len(x.time_seqs)):
        ends = [
            max(b.time_seqs[i][b.valid_event_mask[i]].tolist(), default=0.0)
            for b in (x, y)
        ]
        for b, mask in zip((x, y), masks):
            for j, t in enumerate(b.time_seqs[i].tolist()):
                mask[i, j] = bool(b.valid_event_mask[i, j]) and t <= min(ends)
    return tuple(
        Batch(b.time_seqs, b.time_delta_seqs, b.type_seqs, mask)
        for b, mask in zip((x, y), masks)
    )


def test_each_permutation_restarts_from_raw_data_and_matches_scalar_oracle(monkeypatch):
    rng = np.random.default_rng(18)
    x, y = sample_process(rng, "poisson", size=2), sample_process(
        rng, "poisson", size=2, rate=4.0
    )
    kernel = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 1)
    test = RecomputedTruncationTest(kernel, 24)
    permutations = [torch.tensor(p) for p in itertools.permutations(range(4))]
    iterator = iter(permutations)
    monkeypatch.setattr(torch, "randperm", lambda *a, **kw: next(iterator))
    result = test.compute_statistics(x, y, accumulate=False)
    pool = test._concat_batches(x, y)
    expected = []
    for order in permutations:
        px, py = test._select_batch(pool, order[:2]), test._select_batch(
            pool, order[2:]
        )
        px, py = truncate_oracle(px, py)
        expected.append(test.mmd(px, py))
    torch.testing.assert_close(result["permuted_statistics"], torch.stack(expected))
    torch.testing.assert_close(
        result["observed_statistic"], test.mmd(*truncate_oracle(x, y))
    )
    assert x.valid_event_mask.all() and y.valid_event_mask.all()
    # Full permutation group includes both labels AND within-group orderings:
    # the latter affect the pairing and cannot be omitted from an exact oracle.
    values = torch.stack(expected)
    exact_p = (values[:, None] >= values[None, :]).double().mean(0)
    for alpha in (0.05, 0.1, 0.2, 0.5):
        assert float((exact_p <= alpha).double().mean()) <= alpha + 1e-12


def test_pooled_gram_experiment_matches_actual_kernel_and_both_public_protocols():
    rng = np.random.default_rng(12)
    x, y = sample_process(rng, "hawkes", size=3), sample_process(
        rng, "hawkes", size=3, rate=4.0
    )
    kernel = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 1)
    test = MMDTwoSampleTest(kernel, 19)
    torch.manual_seed(83)
    orders = torch.stack([torch.arange(6)] + [torch.randperm(6) for _ in range(19)])
    fast = paired_statistics(test, x, y, orders)
    pool = test._concat_batches(x, y)
    for index, order in enumerate(orders):
        px, py = test._select_batch(pool, order[:3]), test._select_batch(
            pool, order[3:]
        )
        torch.testing.assert_close(fast["without_truncation"][index], test.mmd(px, py))
        torch.testing.assert_close(
            fast["recomputed_pairwise_truncation"][index],
            test.mmd(*truncate_oracle(px, py)),
        )
    for name, implementation in (
        ("without_truncation", test),
        ("recomputed_pairwise_truncation", RecomputedTruncationTest(kernel, 19)),
    ):
        torch.manual_seed(83)
        if name == "without_truncation":
            observed, null = implementation._permutation_test(x, y)
        else:
            stats = implementation.compute_statistics(x, y, accumulate=False)
            observed, null = stats["observed_statistic"], stats["permuted_statistics"]
        torch.testing.assert_close(fast[name], torch.cat([observed.reshape(1), null]))
        assert count_ge(fast[name][0], fast[name][1:]) == count_ge(observed, null)


def test_experimental_truncation_does_not_silently_broadcast_unpaired_groups():
    rng = np.random.default_rng(3)
    x, y = sample_process(rng, "poisson", size=2), sample_process(
        rng, "poisson", size=3
    )
    kernel = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 1)
    test = RecomputedTruncationTest(kernel, 3)
    with pytest.raises(ValueError, match="equal group sizes"):
        test.compute_statistics(x, y)
    with pytest.raises(ValueError, match="permutations only"):
        test.compute_statistics(x, x, simulations=[x, x])


def test_exact_alpha_boundary_and_numerical_ties_are_conservative():
    observed = torch.tensor(1.0, dtype=torch.float64)
    assert permutation_p_value(observed, torch.zeros(19, dtype=torch.float64)) == 0.05
    null = torch.cat(
        [
            torch.full((4,), 2.0, dtype=torch.float64),
            torch.zeros(95, dtype=torch.float64),
        ]
    )
    assert permutation_p_value(observed, null) == 0.05
    assert (
        count_ge(observed, torch.tensor([1.0 - 1e-14, 0.9], dtype=torch.float64)) == 1
    )
