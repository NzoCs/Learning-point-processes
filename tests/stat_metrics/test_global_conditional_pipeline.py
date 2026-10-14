"""Dense reference, cross-batch terms, identities and trajectory RNG controls."""

import pytest
import torch

from new_ltpp.configs import StatisticalTestConfig
from new_ltpp.evaluation.accumulators.statistical_metrics_accumulator import (
    StatisticalTestAccumulator,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test import (
    ConditionalMMDTwoSampleTest,
    paired_joint_statistic,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.global_conditional_mmd_test import (
    GLOBAL_NULL_RNG_SALT,
    GlobalConditionalMMDTwoSampleTest,
    copy_rows,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)
from new_ltpp.shared_types import Batch


def fixtures(stopped=False):
    def batch(times, masks=None, flags=None):
        times = torch.tensor(times, dtype=torch.float64)
        mask = (
            torch.ones_like(times, dtype=torch.bool)
            if masks is None
            else torch.tensor(masks)
        )
        gaps = torch.zeros_like(times)
        for row in range(len(times)):
            t = times[row][mask[row]]
            gaps[row][mask[row]] = torch.diff(t, prepend=torch.zeros_like(t[:1]))
        return Batch(
            times,
            gaps,
            torch.tensor([[0, 1]] * 5),
            mask,
            no_next_event_mask=None if flags is None else torch.tensor(flags),
            sequence_ids=[f"id-{i}" for i in range(5)],
        )

    h = batch([[0, 0.2], [0, 1], [0, 0.3], [0, 2], [0, 0.6]])
    x = batch([[0.1, 0.2], [0.4, 0.8], [0.3, 1.2], [0.5, 3], [0.2, 0.5]])
    y = batch([[0.5, 2], [0.1, 0.3], [0.8, 1.5], [1, 4], [0.2, 0.4]])
    if stopped:
        x = batch(
            [[0, 0], [0.4, 0], [0.3, 1.2], [0.5, 3], [0.2, 0.5]],
            [[False, False], [True, False], [True, True], [True, True], [True, True]],
            [True, True, False, False, False],
        )
        y.no_next_event_mask = torch.tensor([False, False, False, False, False])
    return h, x, y


def kernel():
    return SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)


@pytest.mark.parametrize("design", ["truncated", "stopped", "window"])
@pytest.mark.parametrize("block_size", [1, 2, 64])
def test_global_blocks_match_dense_every_swap_and_partition(design, block_size):
    torch.manual_seed(120)
    h, x, y = fixtures(design == "stopped")
    kwargs = (
        {"stopped_event_budget": 2}
        if design == "stopped"
        else ({"observation_horizon": 5} if design == "window" else {})
    )
    global_test = GlobalConditionalMMDTwoSampleTest(
        kernel(), 19, block_size=block_size, **kwargs
    )
    dense = ConditionalMMDTwoSampleTest(kernel(), 19, **kwargs).joint_kernel_matrix(
        h, x, y
    )
    generator = torch.Generator().manual_seed((120 + GLOBAL_NULL_RNG_SALT) % (2**63))
    swaps = torch.randint(2, (19, 5), generator=generator)
    rows = torch.arange(5)
    expected = torch.stack(
        [paired_joint_statistic(dense, rows, rows + 5)]
        + [
            paired_joint_statistic(dense, rows + 5 * s, rows + 5 * (1 - s))
            for s in swaps
        ]
    )
    # Different ordering and a singleton tail must not change the null assignment.
    for indices in ([3, 1], [4, 0], [2]):
        ih, ix, iy = (copy_rows(b, indices) for b in (h, x, y))
        global_test.compute_statistics(ix, iy, histories=ih, pair_ids=ih.sequence_ids)
    result = global_test.global_results()
    actual = torch.tensor(
        [result["global_observed_statistic"], *result["global_permuted_statistic"]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual.double(), expected, rtol=1e-10, atol=1e-12)
    assert result["global_pair_ids"] == h.sequence_ids
    assert global_test.get_final_p_value() == monte_carlo_p_value(
        expected[0], expected[1:]
    )
    full = GlobalConditionalMMDTwoSampleTest(
        kernel(), 19, block_size=block_size, **kwargs
    )
    full.compute_statistics(x, y, histories=h, pair_ids=h.sequence_ids)
    assert full.global_results() == result


def test_gram_calls_bounded_and_no_trajectory_rng_consumption(monkeypatch):
    torch.manual_seed(811)
    cfg = StatisticalTestConfig(
        test_type="mmd",
        point_process_kernel_type="sig_kernel",
        space_kernel_type="rbf",
        num_event_types=2,
        n_samples=9,
        num_discretization_points=8,
        signature_max_batch=2,
    )
    acc = StatisticalTestAccumulator(cfg, conditional=True)
    h, x, y = fixtures()
    before = torch.get_rng_state().clone()
    for indices in ([0, 1], [2, 3], [4]):
        ih, ix, iy = (copy_rows(b, indices) for b in (h, x, y))
        acc.update(ix, iy, histories=ih, pair_ids=ih.sequence_ids)
    original = acc.test.kernel.kernel.compute_Gram
    shapes = []

    def checked(a, b):
        shapes.append((len(a), len(b)))
        assert len(a) <= 2 and len(b) <= 2
        return original(a, b)

    monkeypatch.setattr(acc.test.kernel.kernel, "compute_Gram", checked)
    result = acc.compute()
    assert shapes and result["num_sequences"] == 5 and len(result["p_values"]) == 2
    assert torch.equal(before, torch.get_rng_state())
    assert acc.compute() == result
    acc.reset()
    assert not acc.test._batches and not acc.test._seen_pair_ids
    assert acc.compute()["global_observed_statistic"] is None


def test_buffer_snapshot_identity_errors_and_append_invalidates_cache():
    h, x, y = fixtures(True)
    test = GlobalConditionalMMDTwoSampleTest(kernel(), 9, stopped_event_budget=2)

    def add(indices):
        ih, ix, iy = (copy_rows(b, indices) for b in (h, x, y))
        return test.compute_statistics(ix, iy, histories=ih, pair_ids=ih.sequence_ids)

    add([0, 1])
    first = test.global_results()
    with pytest.raises(ValueError, match="Overlapping"):
        add([1, 2])
    with pytest.raises(ValueError, match="stable"):
        test.compute_statistics(x, y, histories=h)
    assert test.global_results() == first
    add([2, 3, 4])
    result = test.global_results()
    x.time_seqs.fill_(123)  # Caller mutation must not alter a buffered observation.
    test.total_observed_mmd = test.total_perm_mmds = None
    assert test.global_results() == result and len(result["global_pair_ids"]) == 5
    test.reset_accumulators()
    h, x, y = fixtures(True)
    add([1])
    with pytest.raises(ValueError, match="in total"):
        test.get_final_p_value()


def test_block_size_and_non_signature_kernel_rejected():
    for size in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            GlobalConditionalMMDTwoSampleTest(kernel(), 9, block_size=size)
    with pytest.raises(ValueError, match="sig_kernel"):
        GlobalConditionalMMDTwoSampleTest(object(), 9)


def test_snapshot_slice_preserves_ordered_ids_and_has_no_storage_alias():
    _, x, _ = fixtures(True)
    selected = copy_rows(x, slice(1, 4))
    assert selected.sequence_ids == x.sequence_ids[1:4]
    assert len(selected.time_seqs) == len(selected.sequence_ids) == 3
    x.time_seqs.fill_(99)
    assert (selected.time_seqs < 99).all()
    assert selected.no_next_event_mask.tolist() == [True, False, False]
