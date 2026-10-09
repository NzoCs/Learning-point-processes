"""Tests for SIGKernel with synthetic data - normalization and embedding."""

import pytest
import torch
import pysiglib

from new_ltpp.evaluation.statistical_testing.point_process_kernels.sig_kernel import (
    SIGKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    LinearKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.utils import (
    _get_embedding,
)
from new_ltpp.evaluation.statistical_testing.point_process_metric import MMD
from new_ltpp.shared_types import Batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_TYPES = 3
BATCH_SIZE = 8
SEQ_LEN = 20
N_DISC = 32  # num_discretization_points kept small for speed


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN, seed: int = 0
) -> Batch:
    """Create a synthetic Batch with realistic inter-event time structure."""
    torch.manual_seed(seed)
    deltas = torch.rand(batch_size, seq_len).abs() + 0.01
    times = torch.cumsum(deltas, dim=1)
    types = torch.randint(0, NUM_TYPES, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return Batch(
        time_seqs=times,
        time_delta_seqs=deltas,
        type_seqs=types,
        valid_event_mask=mask,
    )


def _linear_kernel() -> SIGKernel:
    return SIGKernel(
        static_kernel=LinearKernel(),
        embedding_type="linear",
        num_discretization_points=N_DISC,
        dyadic_order=2,
        num_event_types=NUM_TYPES,
    )


def _constant_kernel() -> SIGKernel:
    return SIGKernel(
        static_kernel=LinearKernel(),
        embedding_type="constant",
        num_discretization_points=N_DISC,
        dyadic_order=2,
        num_event_types=NUM_TYPES,
    )


@pytest.fixture(scope="module")
def batch1():
    return _make_batch(seed=0)


@pytest.fixture(scope="module")
def batch2():
    return _make_batch(seed=42)


# ---------------------------------------------------------------------------
# Tests – embedding utility
# ---------------------------------------------------------------------------


class TestEmbeddingShape:
    """Shape and validity of the embedding produced by _get_embedding."""

    def test_linear_embedding_shape(self, batch1):
        # Normalize times to [0,1] as _prepare_kernel would do
        t = batch1.time_seqs.double()
        t = t / (t.max() + 1e-8)
        emb = _get_embedding(
            N_DISC,
            "linear",
            NUM_TYPES,
            t,
            batch1.type_seqs.long(),
            batch1.valid_event_mask,
        )
        assert emb.shape == (BATCH_SIZE, N_DISC, 2 + NUM_TYPES)

    def test_constant_embedding_shape(self, batch1):
        t = batch1.time_seqs.double()
        t = t / (t.max() + 1e-8)
        emb = _get_embedding(
            N_DISC,
            "constant",
            NUM_TYPES,
            t,
            batch1.type_seqs.long(),
            batch1.valid_event_mask,
        )
        # constant mode still outputs N_DISC points (grid-based)
        assert emb.shape == (BATCH_SIZE, N_DISC, 2 + NUM_TYPES)

    def test_no_nan_linear(self, batch1):
        t = batch1.time_seqs.double()
        t = t / (t.max() + 1e-8)
        emb = _get_embedding(
            N_DISC,
            "linear",
            NUM_TYPES,
            t,
            batch1.type_seqs.long(),
            batch1.valid_event_mask,
        )
        assert not torch.isnan(emb).any()
        assert not torch.isinf(emb).any()

    def test_time_channel_normalized(self, batch1):
        """First channel should be the time grid [0, 1]."""
        t = batch1.time_seqs.double()
        t = t / (t.max() + 1e-8)
        emb = _get_embedding(
            N_DISC,
            "linear",
            NUM_TYPES,
            t,
            batch1.type_seqs.long(),
            batch1.valid_event_mask,
        )
        time_channel = emb[:, :, 0]
        assert torch.all(time_channel >= 0.0)
        assert torch.all(time_channel <= 1.0 + 1e-6)

    def test_counting_channel_normalized(self, batch1):
        """Scale by valid event count, independently of padded batch width."""
        t = batch1.time_seqs.double()
        t = t / (t.max() + 1e-8)
        emb = _get_embedding(
            N_DISC,
            "linear",
            NUM_TYPES,
            t,
            batch1.type_seqs.long(),
            batch1.valid_event_mask,
        )
        counting = emb[:, :, 1]
        assert torch.all(counting >= 0.0)
        valid_count = batch1.valid_event_mask.sum(dim=1).double()
        expected_end = valid_count / ((valid_count - 1).clamp_min(1) + 1e-8)
        torch.testing.assert_close(counting[:, -1], expected_end)
        assert torch.all(counting <= expected_end[:, None] + 1e-6)


# ---------------------------------------------------------------------------
# Tests – Gram matrix
# ---------------------------------------------------------------------------


class TestGramMatrix:
    """Gram matrix shape, symmetry, and numerical validity."""

    def test_gram_shape_linear_kernel(self, batch1, batch2):
        kernel = _linear_kernel()
        gram = kernel.compute_gram_matrix(batch1, batch2)
        assert gram.shape == (BATCH_SIZE, BATCH_SIZE)

    def test_gram_shape_constant_kernel(self, batch1, batch2):
        kernel = _constant_kernel()
        gram = kernel.compute_gram_matrix(batch1, batch2)
        assert gram.shape == (BATCH_SIZE, BATCH_SIZE)

    def test_no_nan_linear(self, batch1, batch2):
        kernel = _linear_kernel()
        gram = kernel.compute_gram_matrix(batch1, batch2)
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

    def test_no_nan_constant(self, batch1, batch2):
        kernel = _constant_kernel()
        gram = kernel.compute_gram_matrix(batch1, batch2)
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

    def test_symmetry_same_batch(self, batch1):
        kernel = _linear_kernel()
        gram = kernel.compute_gram_matrix(batch1, batch1)
        assert torch.allclose(gram, gram.t(), atol=1e-4), "Gram matrix is not symmetric"

    def test_positive_diagonal(self, batch1):
        kernel = _linear_kernel()
        gram = kernel.compute_gram_matrix(batch1, batch1)
        assert torch.all(torch.diag(gram) > 0)

    def test_dtype_preserved(self, batch1, batch2):
        kernel = _linear_kernel()
        gram = kernel.compute_gram_matrix(batch1, batch2)
        # SIGKernel internally uses double; output may be double or float
        assert gram.dtype in (torch.float32, torch.float64)


# ---------------------------------------------------------------------------
# Tests – MMD
# ---------------------------------------------------------------------------


class TestMMDWithSIGKernel:
    """MMD metric via SIG kernel."""

    @pytest.mark.parametrize("embedding", ["linear", "constant"])
    @pytest.mark.parametrize("same_sample", [True, False])
    def test_matches_native_unbiased_mmd(self, batch1, batch2, embedding, same_sample):
        # Unbiased MMD² is not constrained to be non-negative and does not
        # vanish when the very same finite sample is used on both sides.
        kernel = _linear_kernel() if embedding == "linear" else _constant_kernel()
        other = batch1 if same_sample else batch2
        x, y = kernel._prepare_kernel(batch1, other)
        reference = pysiglib.sig_mmd(
            x,
            y,
            method="finite_difference",
            dyadic_order=kernel.dyadic_order,
            static_kernel=kernel.kernel.static_kernel,
        )
        actual = MMD(kernel=kernel)(batch1, other)
        assert torch.isfinite(actual)
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-8)

    def test_mmd_invariant_to_sample_order(self, batch1, batch2):
        indices = torch.arange(BATCH_SIZE - 1, -1, -1)
        reordered = Batch(
            time_seqs=batch2.time_seqs[indices],
            time_delta_seqs=batch2.time_delta_seqs[indices],
            type_seqs=batch2.type_seqs[indices],
            valid_event_mask=batch2.valid_event_mask[indices],
        )
        metric = MMD(kernel=_linear_kernel())
        torch.testing.assert_close(
            metric(batch1, batch2), metric(batch1, reordered), rtol=1e-9, atol=1e-8
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
