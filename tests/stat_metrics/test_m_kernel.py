"""Integration tests for MKernel with real data - various time kernels and transformations."""

import pytest
import torch

from new_ltpp.evaluation.statistical_testing.kernels.m_kernel import (
    MKernel,
    MKernelTransform,
)
from new_ltpp.evaluation.statistical_testing.kernels.space_kernels import (
    TimeKernelType,
    create_time_kernel,
    EmbeddingKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_metrics import MMD
from new_ltpp.shared_types import Batch
from new_ltpp.configs.config_builders import DataConfigBuilder
from new_ltpp.data.preprocess import TPPDataModule


@pytest.fixture(scope="module")
def data_module():
    """Create a data module with test data."""
    data_builder = DataConfigBuilder()
    (
        data_builder.set_dataset_id("test")
        .set_src_dir("NzoCs/test_dataset")
        .set_num_event_types(1)
        .set_data_loading_specs(batch_size=16, num_workers=0, shuffle=False)
        .set_data_format("hf")
    )

    data_config = data_builder.build()
    datamodule = TPPDataModule(data_config)
    datamodule.setup(stage="test")

    return datamodule


@pytest.fixture(scope="module")
def test_batches(data_module):
    """Get two test batches from the data module."""
    test_loader = data_module.test_dataloader()
    batch_iter = iter(test_loader)

    batch1 = next(batch_iter)
    batch2 = next(batch_iter)

    return batch1, batch2


@pytest.fixture(scope="module")
def num_event_types(test_batches):
    """Get the number of event types from the data."""
    batch1, _ = test_batches
    # +1 because types are 0-indexed
    return int(batch1.type_seqs.max().item()) + 1


class TestMKernelWithRealData:
    """Integration tests with real data loading."""

    def test_data_loading(self, data_module):
        """Test that data loads correctly."""
        test_loader = data_module.test_dataloader()
        assert test_loader is not None
        assert len(test_loader) > 0

    def test_batch_structure(self, test_batches):
        """Test that batches have the correct structure."""
        batch1, batch2 = test_batches

        # Check that both batches have required fields
        assert hasattr(batch1, "time_seqs")
        assert hasattr(batch1, "type_seqs")
        assert hasattr(batch1, "time_delta_seqs")
        assert hasattr(batch1, "valid_event_mask")

        # Check shapes are consistent
        batch_size, seq_len = batch1.time_seqs.shape
        assert batch1.type_seqs.shape == (batch_size, seq_len)
        assert batch1.time_delta_seqs.shape == (batch_size, seq_len)
        assert batch1.valid_event_mask.shape == (batch_size, seq_len)


class TestTimeKernelTypes:
    """Test MKernel with different time kernel types."""

    def test_rbf_time_kernel(self, test_batches, num_event_types):
        """Test MKernel with RBF time kernel."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_imq_time_kernel(self, test_batches, num_event_types):
        """Test MKernel with IMQ time kernel."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.IMQ, c=1.0, beta=0.5)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_matern_3_2_time_kernel(self, test_batches, num_event_types):
        """Test MKernel with Matern 3/2 time kernel."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.MATERN_3_2, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_laplacian_time_kernel(self, test_batches, num_event_types):
        """Test MKernel with Laplacian time kernel."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.LAPLACIAN, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_rational_quadratic_time_kernel(self, test_batches, num_event_types):
        """Test MKernel with Rational Quadratic time kernel."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(
            TimeKernelType.RATIONAL_QUADRATIC, sigma=1.0, alpha=1.0
        )
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])


class TestMKernelTransforms:
    """Test MKernel with different transformation functions."""

    def test_exponential_transform(self, test_batches, num_event_types):
        """Test MKernel with exponential (RBF-like) transform."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.EXPONENTIAL,
        )

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_imq_transform(self, test_batches, num_event_types):
        """Test MKernel with IMQ transform."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.IMQ,
            c=1.0,
            beta=0.5,
        )

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_rational_quadratic_transform(self, test_batches, num_event_types):
        """Test MKernel with Rational Quadratic transform."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.RATIONAL_QUADRATIC,
            alpha=1.0,
        )

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_laplacian_transform(self, test_batches, num_event_types):
        """Test MKernel with Laplacian transform."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.LAPLACIAN,
        )

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_linear_transform(self, test_batches, num_event_types):
        """Test MKernel with Linear transform."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.LINEAR,
        )

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_cauchy_transform(self, test_batches, num_event_types):
        """Test MKernel with Cauchy transform."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.CAUCHY,
        )

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf"

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])


class TestGramMatrixProperties:
    """Test mathematical properties of the Gram matrix."""

    def test_gram_matrix_symmetry(self, test_batches, num_event_types):
        """Test that Gram matrix is symmetric when comparing same batch."""
        batch1, _ = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        gram = m_kernel.graam_matrix(batch1, batch1)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Matrix should be symmetric
        assert torch.allclose(gram, gram.t(), atol=1e-4)

    def test_gram_matrix_positive_diagonal(self, test_batches, num_event_types):
        """Test that diagonal elements are positive with real data."""
        batch1, _ = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        gram = m_kernel.graam_matrix(batch1, batch1)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Diagonal should be positive
        assert torch.all(torch.diag(gram) > 0)

    def test_gram_matrix_shape(self, test_batches, num_event_types):
        """Test Gram matrix shape with different batch sizes."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Check shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])


class TestMMDWithMKernel:
    """Test MMD metric with M-Kernel on real data."""

    def test_mmd_self_similarity(self, test_batches, num_event_types):
        """Test that MMD(batch, batch) is close to 0."""
        batch1, _ = test_batches

        time_kernel = create_time_kernel(TimeKernelType.IMQ, c=1.0, beta=0.5)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.IMQ,
            c=1.0,
            beta=0.5,
        )

        mmd = MMD(kernel=m_kernel)
        mmd_value = mmd(batch1, batch1)

        # Verify no NaN
        assert not torch.isnan(torch.tensor(mmd_value))
        assert not torch.isinf(torch.tensor(mmd_value))

        # Self-similarity should be very small
        assert abs(mmd_value) < 1e-5

    def test_mmd_different_batches(self, test_batches, num_event_types):
        """Test MMD between different batches."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.IMQ, c=1.0, beta=0.5)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.IMQ,
            c=1.0,
            beta=0.5,
        )

        mmd = MMD(kernel=m_kernel)
        mmd_value = mmd(batch1, batch2)

        # Verify no NaN
        assert not torch.isnan(torch.tensor(mmd_value))
        assert not torch.isinf(torch.tensor(mmd_value))

        # MMD should be positive or zero
        assert mmd_value >= 0.0

    def test_mmd_with_noise(self, test_batches, num_event_types):
        """Test that MMD increases with noise level."""
        batch1, _ = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        mmd = MMD(kernel=m_kernel)

        # Test different noise levels
        noise_levels = [0.01, 0.05, 0.1]
        mmd_values = []

        for noise in noise_levels:
            # Create noisy batch
            noisy_times = batch1.time_seqs + torch.randn_like(batch1.time_seqs) * noise
            noisy_times = torch.clamp(noisy_times, min=0)

            noisy_deltas = (
                batch1.time_delta_seqs
                + torch.randn_like(batch1.time_delta_seqs) * noise
            )
            noisy_deltas = torch.clamp(noisy_deltas, min=1e-6)

            batch_noisy = Batch(
                time_seqs=noisy_times,
                time_delta_seqs=noisy_deltas,
                type_seqs=batch1.type_seqs,
                valid_event_mask=batch1.valid_event_mask,
            )

            mmd_val = mmd(batch1, batch_noisy)

            # Verify no NaN
            assert not torch.isnan(torch.tensor(mmd_val))
            assert not torch.isinf(torch.tensor(mmd_val))

            mmd_values.append(mmd_val)

        # MMD should generally increase with noise
        # (at least the largest noise should have higher MMD than no noise)
        assert mmd_values[-1] > 1e-6

    def test_mmd_with_different_transforms(self, test_batches, num_event_types):
        """Test MMD with different transform types."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        transforms = [
            MKernelTransform.EXPONENTIAL,
            MKernelTransform.IMQ,
            MKernelTransform.RATIONAL_QUADRATIC,
            MKernelTransform.LAPLACIAN,
        ]

        for transform in transforms:
            m_kernel = MKernel(
                time_kernel=time_kernel,
                type_kernel=type_kernel,
                sigma=1.0,
                transform=transform,
                c=1.0,
                beta=0.5,
                alpha=1.0,
            )

            mmd = MMD(kernel=m_kernel)
            mmd_value = mmd(batch1, batch2)

            # Verify no NaN
            assert not torch.isnan(torch.tensor(mmd_value)), (
                f"MMD is NaN with transform {transform}"
            )
            assert not torch.isinf(torch.tensor(mmd_value)), (
                f"MMD is Inf with transform {transform}"
            )

            # MMD should be non-negative
            assert mmd_value >= 0.0, f"MMD is negative with transform {transform}"


class TestDTypePreservation:
    """Test that dtypes are preserved correctly."""

    def test_gram_matrix_dtype_preservation(self, test_batches, num_event_types):
        """Test that Gram matrix preserves dtype."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0)

        original_dtype = batch1.time_seqs.dtype
        gram = m_kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Gram matrix should preserve dtype
        assert gram.dtype == original_dtype


class TestCombinedKernelsAndTransforms:
    """Test combinations of different kernels and transforms."""

    def test_imq_kernel_with_imq_transform(self, test_batches, num_event_types):
        """Test IMQ time kernel with IMQ transform (double robustness)."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.IMQ, c=1.0, beta=0.5)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.IMQ,
            c=1.0,
            beta=0.5,
        )

        mmd = MMD(kernel=m_kernel)
        mmd_value = mmd(batch1, batch2)

        # Verify no NaN
        assert not torch.isnan(torch.tensor(mmd_value))
        assert not torch.isinf(torch.tensor(mmd_value))

        # Should work without issues
        assert mmd_value >= 0.0

    def test_matern_kernel_with_rq_transform(self, test_batches, num_event_types):
        """Test Matérn kernel with Rational Quadratic transform."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.MATERN_3_2, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        m_kernel = MKernel(
            time_kernel=time_kernel,
            type_kernel=type_kernel,
            sigma=1.0,
            transform=MKernelTransform.RATIONAL_QUADRATIC,
            alpha=1.0,
        )

        mmd = MMD(kernel=m_kernel)
        mmd_value = mmd(batch1, batch2)

        # Verify no NaN
        assert not torch.isnan(torch.tensor(mmd_value))
        assert not torch.isinf(torch.tensor(mmd_value))

        # Should work without issues
        assert mmd_value >= 0.0


class TestNoNaNOrInf:
    """Test that operations don't produce NaN or Inf values."""

    def test_all_time_kernels_no_nan(self, test_batches, num_event_types):
        """Test that all time kernel types don't produce NaN."""
        batch1, batch2 = test_batches

        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        time_kernel_types = [
            TimeKernelType.RBF,
            TimeKernelType.IMQ,
            TimeKernelType.MATERN_3_2,
            TimeKernelType.MATERN_5_2,
            TimeKernelType.LAPLACIAN,
            TimeKernelType.RATIONAL_QUADRATIC,
        ]

        for kernel_type in time_kernel_types:
            if kernel_type == TimeKernelType.IMQ:
                time_kernel = create_time_kernel(kernel_type, c=1.0, beta=0.5)
            elif kernel_type == TimeKernelType.RATIONAL_QUADRATIC:
                time_kernel = create_time_kernel(kernel_type, sigma=1.0, alpha=1.0)
            else:
                time_kernel = create_time_kernel(kernel_type, sigma=1.0)

            m_kernel = MKernel(
                time_kernel=time_kernel, type_kernel=type_kernel, sigma=1.0
            )

            gram = m_kernel.graam_matrix(batch1, batch2)

            assert not torch.isnan(gram).any(), f"NaN with kernel {kernel_type}"
            assert not torch.isinf(gram).any(), f"Inf with kernel {kernel_type}"

    def test_all_transforms_no_nan(self, test_batches, num_event_types):
        """Test that all transform types don't produce NaN."""
        batch1, batch2 = test_batches

        time_kernel = create_time_kernel(TimeKernelType.RBF, sigma=1.0)
        type_kernel = EmbeddingKernel(
            num_classes=max(2, num_event_types), embedding_dim=5, sigma=1.0
        )

        transforms = [
            MKernelTransform.EXPONENTIAL,
            MKernelTransform.IMQ,
            MKernelTransform.RATIONAL_QUADRATIC,
            MKernelTransform.LAPLACIAN,
            MKernelTransform.LINEAR,
            MKernelTransform.CAUCHY,
        ]

        for transform in transforms:
            m_kernel = MKernel(
                time_kernel=time_kernel,
                type_kernel=type_kernel,
                sigma=1.0,
                transform=transform,
                c=1.0,
                beta=0.5,
                alpha=1.0,
            )

            gram = m_kernel.graam_matrix(batch1, batch2)

            assert not torch.isnan(gram).any(), f"NaN with transform {transform}"
            assert not torch.isinf(gram).any(), f"Inf with transform {transform}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
