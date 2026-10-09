"""Independent counting oracles and padding invariance contracts."""

import pytest
import torch

from new_ltpp.shared_types import Batch
from new_ltpp.evaluation.statistical_testing.point_process_kernels.utils import (
    _get_embedding,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.sig_kernel import (
    SIGKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    LinearKernel,
)


def sample(padding=None):
    times = [0.25, 0.75]
    marks = [0, 1]
    mask = [True, True]
    if padding == "left":
        times = [99.0, 99.0] + times
        marks = [-1, -1] + marks
        mask = [False, False] + mask
    elif padding == "right":
        times += [99.0, 99.0]
        marks += [-1, -1]
        mask += [False, False]
    t = torch.tensor([times], dtype=torch.float64)
    return Batch(t, torch.zeros_like(t), torch.tensor([marks]), torch.tensor([mask]))


def embedding(batch):
    return _get_embedding(
        5, "linear", 2, batch.time_seqs, batch.type_seqs, batch.valid_event_mask
    )


def test_counting_matches_hand_calculated_events_at_grid_boundaries():
    # Events at 1/4 and 3/4 are included at those grid points (N(t) counts <= t).
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 1.0, 1.0, 0.0],
            [0.5, 1.0, 1.0, 0.0],
            [0.75, 2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)
    torch.testing.assert_close(embedding(sample()), expected, rtol=1e-7, atol=1e-7)


def test_tied_events_and_empty_sequence_match_hand_counts():
    times = torch.tensor([[0.5, 0.5, 0.75], [99.0, 99.0, 99.0]], dtype=torch.float64)
    marks = torch.tensor([[0, 1, 0], [-1, -1, -1]])
    mask = torch.tensor([[True, True, True], [False, False, False]])
    actual = embedding(Batch(times, torch.zeros_like(times), marks, mask))
    expected_counts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.5, 2.0, 1.0],
            [1.5, 2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual[0, :, 1:], expected_counts, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(actual[1, :, 1:], torch.zeros_like(expected_counts))
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("padding", ["left", "right"])
def test_embedding_is_invariant_to_masked_padding(padding):
    torch.testing.assert_close(embedding(sample(padding)), embedding(sample()))


@pytest.mark.parametrize("padding", ["left", "right"])
def test_signature_gram_is_invariant_to_masked_padding(padding):
    kernel = SIGKernel(LinearKernel(), "linear", 5, 0, 2)
    reference = sample()
    expected = kernel.compute_gram_matrix(reference, reference)
    actual = kernel.compute_gram_matrix(sample(padding), reference)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("padding", [None, "left", "right"])
def test_preparation_and_gram_do_not_mutate_input_batches(padding):
    batch = sample(padding)
    before = {name: tensor.clone() for name, tensor in batch.to_mapping().items()}
    kernel = SIGKernel(LinearKernel(), "linear", 5, 0, 2)
    x, y = kernel._prepare_kernel(batch, batch)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    kernel.compute_gram_matrix(batch, batch)
    for name, tensor in batch.to_mapping().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_unsorted_times_and_interspersed_padding_keep_marks_aligned():
    times = torch.tensor([[0.75, float("nan"), 0.25, 99.0]], dtype=torch.float64)
    batch = Batch(
        times,
        torch.zeros_like(times),
        torch.tensor([[1, -1, 0, -1]]),
        torch.tensor([[True, False, True, False]]),
    )
    torch.testing.assert_close(embedding(batch), embedding(sample()))


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_event_paths_have_finite_hand_counts(count):
    times = torch.tensor([[0.25, 99.0]], dtype=torch.float64)
    batch = Batch(
        times,
        torch.zeros_like(times),
        torch.tensor([[1, -1]]),
        torch.tensor([[bool(count), False]]),
    )
    actual = embedding(batch)
    expected = torch.zeros((5, 3), dtype=torch.float64)
    if count:
        expected[1:, 0] = 1.0
        expected[1:, 2] = 1.0
    torch.testing.assert_close(actual[0, :, 1:], expected, rtol=1e-7, atol=1e-7)
    assert torch.isfinite(actual).all()


def test_longer_batch_companion_does_not_change_short_path():
    # Keep the time horizon fixed: this tests batch width, not time rescaling.
    times = torch.tensor(
        [[0.25, 0.75, 99.0, 99.0], [0.1, 0.2, 0.5, 0.75]], dtype=torch.float64
    )
    batch = Batch(
        times,
        torch.zeros_like(times),
        torch.tensor([[0, 1, -1, -1], [1, 0, 1, 0]]),
        torch.tensor([[True, True, False, False], [True, True, True, True]]),
    )
    kernel = SIGKernel(LinearKernel(), "linear", 5, 0, 2)
    expected_path, _ = kernel._prepare_kernel(sample(), sample())
    actual_path, _ = kernel._prepare_kernel(batch, sample())
    torch.testing.assert_close(actual_path[:1], expected_path)
    torch.testing.assert_close(
        kernel.compute_gram_matrix(batch, sample())[:1],
        kernel.compute_gram_matrix(sample(), sample()),
    )


@pytest.mark.parametrize("padding", ["left", "right"])
def test_native_signature_path_gradients_are_padding_invariant(padding):
    kernel = SIGKernel(LinearKernel(), "linear", 5, 0, 2)

    def calculate(batch):
        x, y = kernel._prepare_kernel(batch, sample())
        # The counting discretization is not differentiable in event times.
        # Check the native solver's gradients with respect to its path inputs.
        x = x.detach().requires_grad_()
        y = y.detach().requires_grad_()
        value = kernel.kernel.compute_Gram(x, y).sum()
        return value.detach(), torch.autograd.grad(value, (x, y))

    expected_value, expected_gradients = calculate(sample())
    actual_value, actual_gradients = calculate(sample(padding))
    torch.testing.assert_close(actual_value, expected_value, rtol=1e-9, atol=1e-9)
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("padding", ["left", "right"])
def test_signature_mmd_with_unequal_samples_is_padding_invariant(padding):
    def repeat(batch, count):
        return Batch(
            *(tensor.repeat(count, 1) for tensor in batch.to_mapping().values())
        )

    kernel = SIGKernel(LinearKernel(), "linear", 5, 0, 2)
    other = repeat(sample(), 3)
    expected = kernel.compute_mmd(repeat(sample(), 2), other)
    actual = kernel.compute_mmd(repeat(sample(padding), 2), other)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
