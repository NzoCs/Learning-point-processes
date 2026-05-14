"""Tests for MKernel with synthetic data - various time kernels and transforms."""

import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels.m_kernel import (
    MKernel,
    MKernelTransform,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    TimeKernelType,
    EmbeddingKernel,
    RBFTimeKernel,
    IMQTimeKernel,
    MaternTimeKernel,
    LaplacianTimeKernel,
    RationalQuadraticTimeKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_metric import MMD
from new_ltpp.shared_types import Batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_TYPES = 3
BATCH_SIZE = 8
SEQ_LEN = 20


def _make_batch(
    batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN, seed: int = 0
) -> Batch:
    """Create a synthetic Batch with realistic inter-event time structure."""
    torch.manual_seed(seed)
    # Strictly increasing times via cumulative sum of exponentials
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


def _make_kernel(
    kernel_type: TimeKernelType = TimeKernelType.RBF,
    transform: MKernelTransform = MKernelTransform.EXPONENTIAL,
) -> MKernel:
    """Instantiate an MKernel for a given time-kernel type and transform."""
    if kernel_type == TimeKernelType.RBF:
        time_kernel = RBFTimeKernel(sigma=1.0)
    elif kernel_type == TimeKernelType.IMQ:
        time_kernel = IMQTimeKernel(c=1.0, beta=0.5)
    elif kernel_type == TimeKernelType.MATERN_3_2:
        time_kernel = MaternTimeKernel(nu=1.5)
    elif kernel_type == TimeKernelType.MATERN_5_2:
        time_kernel = MaternTimeKernel(nu=2.5)
    elif kernel_type == TimeKernelType.LAPLACIAN:
        time_kernel = LaplacianTimeKernel(scaling=1.0)
    elif kernel_type == TimeKernelType.RATIONAL_QUADRATIC:
        time_kernel = RationalQuadraticTimeKernel(sigma=1.0, alpha=1.0)
    else:
        time_kernel = RBFTimeKernel(sigma=1.0)

    type_kernel = EmbeddingKernel(num_classes=NUM_TYPES, embedding_dim=8)
    return MKernel(
        time_kernel=time_kernel,
        type_kernel=type_kernel,
        transform=transform,
        c=1.0,
        beta=0.5,
        alpha=1.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def batch1():
    return _make_batch(seed=0)


@pytest.fixture(scope="module")
def batch2():
    return _make_batch(seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTimeKernelTypes:
    """MKernel with every supported time-kernel type."""

    @pytest.mark.parametrize(
        "ktype",
        [
            TimeKernelType.RBF,
            TimeKernelType.IMQ,
            TimeKernelType.MATERN_3_2,
            TimeKernelType.MATERN_5_2,
            TimeKernelType.LAPLACIAN,
            TimeKernelType.RATIONAL_QUADRATIC,
        ],
    )
    def test_gram_matrix_shape_and_validity(self, batch1, batch2, ktype):
        m_kernel = _make_kernel(ktype)
        gram = m_kernel.compute_gram_matrix(batch1, batch2)
        assert not torch.isnan(gram).any(), f"NaN with kernel {ktype}"
        assert not torch.isinf(gram).any(), f"Inf with kernel {ktype}"
        assert gram.shape == (BATCH_SIZE, BATCH_SIZE)


class TestMKernelTransforms:
    """MKernel with every transform option."""

    @pytest.mark.parametrize("transform", list(MKernelTransform))
    def test_gram_matrix_transform(self, batch1, batch2, transform):
        m_kernel = _make_kernel(transform=transform)
        gram = m_kernel.compute_gram_matrix(batch1, batch2)
        assert not torch.isnan(gram).any(), f"NaN with transform {transform}"
        assert not torch.isinf(gram).any(), f"Inf with transform {transform}"
        assert gram.shape == (BATCH_SIZE, BATCH_SIZE)


class TestGramMatrixProperties:
    """Mathematical properties of the Gram matrix."""

    def test_symmetry_same_batch(self, batch1):
        m_kernel = _make_kernel()
        gram = m_kernel.compute_gram_matrix(batch1, batch1)
        assert torch.allclose(gram, gram.t(), atol=1e-4), "Gram matrix is not symmetric"

    def test_positive_diagonal(self, batch1):
        m_kernel = _make_kernel()
        gram = m_kernel.compute_gram_matrix(batch1, batch1)
        assert torch.all(torch.diag(gram) > 0), "Diagonal elements are not all positive"

    def test_shape(self, batch1, batch2):
        m_kernel = _make_kernel()
        gram = m_kernel.compute_gram_matrix(batch1, batch2)
        assert gram.shape == (BATCH_SIZE, BATCH_SIZE)


class TestMMDWithMKernel:
    """MMD metric computed via M-Kernel."""

    def test_mmd_self_similarity(self, batch1):
        m_kernel = _make_kernel(TimeKernelType.IMQ, MKernelTransform.IMQ)
        mmd = MMD(kernel=m_kernel)
        value = mmd(batch1, batch1)
        assert not torch.isnan(torch.tensor(value))
        assert abs(value) < 1e-5, f"Self-MMD should be ~0, got {value}"

    def test_mmd_non_negative(self, batch1, batch2):
        m_kernel = _make_kernel()
        mmd = MMD(kernel=m_kernel)
        value = mmd(batch1, batch2)
        assert not torch.isnan(torch.tensor(value))
        assert value >= 0.0, f"MMD should be non-negative, got {value}"

    def test_mmd_increases_with_noise(self, batch1):
        m_kernel = _make_kernel()
        mmd = MMD(kernel=m_kernel)
        mmd_values = []
        for noise in [0.01, 0.05, 0.1]:
            noisy_deltas = (
                batch1.time_delta_seqs
                + torch.randn_like(batch1.time_delta_seqs) * noise
            ).clamp(min=1e-6)
            noisy_times = torch.cumsum(noisy_deltas, dim=1)
            noisy_batch = Batch(
                time_seqs=noisy_times,
                time_delta_seqs=noisy_deltas,
                type_seqs=batch1.type_seqs,
                valid_event_mask=batch1.valid_event_mask,
            )
            mmd_values.append(mmd(batch1, noisy_batch))
        assert mmd_values[-1] > 1e-6, "MMD should grow with noise"

    @pytest.mark.parametrize(
        "transform",
        [
            MKernelTransform.EXPONENTIAL,
            MKernelTransform.IMQ,
            MKernelTransform.RATIONAL_QUADRATIC,
            MKernelTransform.LAPLACIAN,
        ],
    )
    def test_mmd_with_transforms(self, batch1, batch2, transform):
        m_kernel = _make_kernel(transform=transform)
        mmd = MMD(kernel=m_kernel)
        value = mmd(batch1, batch2)
        assert not torch.isnan(torch.tensor(value)), (
            f"MMD is NaN with transform {transform}"
        )
        assert value >= 0.0, f"MMD is negative with transform {transform}"


class TestDTypePreservation:
    """Gram matrix should preserve input dtype."""

    def test_dtype_float32(self, batch1, batch2):
        m_kernel = _make_kernel()
        gram = m_kernel.compute_gram_matrix(batch1, batch2)
        assert gram.dtype == batch1.time_seqs.dtype


class TestCombinedKernelsAndTransforms:
    """Representative kernel × transform combinations."""

    def test_imq_kernel_imq_transform(self, batch1, batch2):
        m_kernel = _make_kernel(TimeKernelType.IMQ, MKernelTransform.IMQ)
        mmd = MMD(kernel=m_kernel)
        value = mmd(batch1, batch2)
        assert not torch.isnan(torch.tensor(value))
        assert value >= 0.0

    def test_matern_kernel_rq_transform(self, batch1, batch2):
        m_kernel = _make_kernel(
            TimeKernelType.MATERN_3_2, MKernelTransform.RATIONAL_QUADRATIC
        )
        mmd = MMD(kernel=m_kernel)
        value = mmd(batch1, batch2)
        assert not torch.isnan(torch.tensor(value))
        assert value >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
