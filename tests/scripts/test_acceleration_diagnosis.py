"""Controls for a diagnostic study, independent of its scientific outcome."""

import numpy as np
import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
)
from scripts.acceleration_diagnosis import (
    VARIANTS,
    concat,
    event_paths,
    future_gram,
    history_weights,
    mechanism,
    quota_view,
    speed_view,
    statistics,
)
from scripts.global_signature_calibration import delta_statistics
from scripts.pipeline_protocol_calibration import from_arrays, histories_for_trial


def example():
    batch = from_arrays(
        [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06], [0.3, 0.7, 1.0, 2.0, 3.0, 40.0]],
        [[0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0]],
    )
    batch.no_next_event_mask = torch.zeros(2, dtype=torch.bool)
    return batch


def test_quota_does_not_reveal_later_stop_and_requires_complete_observation():
    batch = example()
    batch.valid_event_mask[0, 4:] = False
    batch.no_next_event_mask[0] = True
    assert quota_view(batch, 3).no_next_event_mask.tolist() == [False, False]
    assert quota_view(batch, 6).no_next_event_mask.tolist() == [True, False]
    batch.no_next_event_mask[0] = False
    with pytest.raises(ValueError, match="Insufficient trajectory"):
        quota_view(batch, 6)


def test_event_paths_retain_late_events_under_fixed_scale_and_no_input_mutation():
    batch = example()
    original = batch.time_seqs.clone()
    paths = event_paths(batch, 1.0)
    assert paths[1, -1, 0] == 40
    torch.testing.assert_close(
        paths[:, -1, 2:], torch.tensor([[3.0, 3.0], [3.0, 3.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(batch.time_seqs, original, rtol=0, atol=0)
    assert torch.all(paths[:, 1:, 0] >= paths[:, :-1, 0])


def test_pooled_grid_can_erase_a_real_time_change_but_event_paths_do_not():
    batch = example()
    kernel = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)
    diagnosis = mechanism(kernel, batch, batch)
    assert diagnosis["acceleration_erased_rows"] >= 1
    old = event_paths(batch, 40.0)
    new = event_paths(speed_view(batch), 40.0)
    assert not torch.equal(old[0], new[0])


def test_all_variants_swap_equivariance_and_reference_statistics():
    x, y = quota_view(example(), 3), quota_view(speed_view(example()), 3)
    h = histories_for_trial(2, np.random.default_rng(7), 0)
    kernel = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)
    weights = history_weights(kernel, h)
    signs = np.asarray([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
    for variant in VARIANTS:
        stats, _, delta = statistics(kernel, weights, x, y, signs, variant, 0.9)
        swapped, _, _ = statistics(kernel, weights, y, x, signs, variant, 0.9)
        np.testing.assert_allclose(stats, swapped, rtol=1e-10, atol=1e-12)
        if variant == "grid8_pool":
            public = (
                ConditionalMMDTwoSampleTest(kernel, 2, stopped_event_budget=3)
                .joint_kernel_matrix(h, x, y)
                .numpy()
            )
            expected = public[:2, :2] + public[2:, 2:] - public[:2, 2:] - public[2:, :2]
            np.testing.assert_allclose(delta, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                stats, delta_statistics(expected, signs), rtol=1e-12, atol=1e-12
            )


def test_compressed_signature_gram_equals_direct_and_time_rescale_preserves_counts():
    batch = concat(example(), example())
    kernel = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)
    gram, unique = future_gram(kernel, batch, "grid8_pool", 1.0)
    expected = kernel.compute_gram_matrix(batch, batch).numpy()
    np.testing.assert_allclose(gram, expected, rtol=1e-12, atol=1e-12)
    assert unique == 2
    # Repeated paths retain multiplicities, including off-diagonal contributions.
    assert gram.shape == (4, 4)


def test_counts_control_is_identical_before_after_acceleration_and_single_pair_swap():
    x, y = quota_view(example(), 3), quota_view(speed_view(example()), 3)
    h = histories_for_trial(2, np.random.default_rng(8), 0)
    kernel = SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)
    weights = history_weights(kernel, h)
    signs = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    before, _, _ = statistics(kernel, weights, x, y, signs, "counts", 0.9)
    after, _, _ = statistics(kernel, weights, speed_view(x), y, signs, "counts", 0.9)
    np.testing.assert_array_equal(before, after)
    # Relabel just one history's pair: preprocessing must commute with it.
    from new_ltpp.evaluation.statistical_testing.statistical_tests.global_conditional_mmd_test import (
        copy_rows,
    )

    pool = concat(x, y)
    swapped_x = copy_rows(pool, [2, 1])
    swapped_y = copy_rows(pool, [0, 3])
    for variant in VARIANTS:
        _, _, delta = statistics(kernel, weights, x, y, signs, variant, 0.9)
        _, _, swapped = statistics(
            kernel, weights, swapped_x, swapped_y, signs, variant, 0.9
        )
        row_sign = np.array([-1.0, 1.0])
        np.testing.assert_allclose(
            swapped,
            delta * row_sign[:, None] * row_sign[None, :],
            rtol=1e-10,
            atol=1e-12,
        )
