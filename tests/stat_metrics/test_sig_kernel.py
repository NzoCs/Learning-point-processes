"""Integration tests for SIGKernel with real data - normalization and one-hot encoding."""

import pytest
import torch

from new_ltpp.evaluation.statistical_testing.kernels.sig_kernel import SIGKernel
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


class TestSIGKernelWithRealData:
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


class TestLinearInterpolantWithRealData:
    """Test linear interpolant embedding with real data."""

    def test_embedding_shape_real_data(self, test_batches, num_event_types):
        """Test embedding shape with real data."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)

        batch_size, seq_len = batch1.time_seqs.shape
        expected_dim = 2 + max(2, num_event_types)
        assert embedding.shape == (batch_size, seq_len, expected_dim)

        # Verify no NaN or Inf
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()

    def test_time_normalization_real_data(self, test_batches, num_event_types):
        """Test that real times are normalized to [0, 1]."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)
        normalized_times = embedding[:, :, 0]

        # Verify no NaN or Inf
        assert not torch.isnan(normalized_times).any()
        assert not torch.isinf(normalized_times).any()

        # Check normalization
        assert torch.all(normalized_times >= 0.0)
        assert torch.all(normalized_times <= 1.0 + 1e-6)  # Allow small numerical error

        # Check that max times are close to 1.0
        max_times_per_seq = normalized_times.max(dim=1)[0]
        assert torch.allclose(
            max_times_per_seq, torch.ones_like(max_times_per_seq), atol=1e-5
        )

    def test_counting_normalization_real_data(self, test_batches, num_event_types):
        """Test that counting sequences are normalized to [0, 1]."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)
        normalized_counting = embedding[:, :, 1]

        # Verify no NaN or Inf
        assert not torch.isnan(normalized_counting).any()
        assert not torch.isinf(normalized_counting).any()

        # Check normalization
        assert torch.all(normalized_counting >= 0.0)
        assert torch.all(normalized_counting <= 1.0 + 1e-6)

        # First event should be 0, last should be 1
        assert torch.allclose(
            normalized_counting[:, 0], torch.zeros(batch1.time_seqs.shape[0]), atol=1e-6
        )
        assert torch.allclose(
            normalized_counting[:, -1], torch.ones(batch1.time_seqs.shape[0]), atol=1e-6
        )

    def test_one_hot_encoding_real_data(self, test_batches, num_event_types):
        """Test that type sequences are one-hot encoded correctly with real data."""
        batch1, _ = test_batches

        num_classes = max(2, num_event_types)
        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=num_classes,
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)
        one_hot = embedding[:, :, 2:]

        # Check that each position has exactly one 1 and rest are 0
        sums = one_hot.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

        # Check that values are either 0 or 1
        assert torch.all((one_hot == 0.0) | (one_hot == 1.0))


class TestConstantInterpolantWithRealData:
    """Test constant interpolant embedding with real data."""

    def test_embedding_shape_real_data(self, test_batches, num_event_types):
        """Test that embedding has doubled sequence length with real data."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="constant_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)

        batch_size, seq_len = batch1.time_seqs.shape
        expected_dim = 2 + max(2, num_event_types)
        assert embedding.shape == (batch_size, 2 * seq_len, expected_dim)

        # Verify no NaN or Inf
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()

    def test_time_duplication_real_data(self, test_batches, num_event_types):
        """Test that times are duplicated correctly with real data."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="constant_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)
        normalized_times = embedding[:, :, 0]

        # Check that times are duplicated (each pair should be identical)
        seq_len = batch1.time_seqs.shape[1]
        for i in range(seq_len):
            assert torch.allclose(
                normalized_times[:, 2 * i], normalized_times[:, 2 * i + 1], atol=1e-6
            )

    def test_one_hot_duplication_real_data(self, test_batches, num_event_types):
        """Test that one-hot vectors are duplicated correctly with real data."""
        batch1, _ = test_batches

        num_classes = max(2, num_event_types)
        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="constant_interpolant",
            dyadic_order=2,
            num_event_types=num_classes,
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)
        one_hot = embedding[:, :, 2:]

        # Check that one-hot vectors are duplicated
        seq_len = batch1.time_seqs.shape[1]
        for i in range(seq_len):
            assert torch.allclose(
                one_hot[:, 2 * i, :], one_hot[:, 2 * i + 1, :], atol=1e-6
            )


class TestGramMatrixWithRealData:
    """Test Gram matrix computation with real data."""

    def test_gram_matrix_shape_real_data(self, test_batches, num_event_types):
        """Test Gram matrix shape with real data."""
        batch1, batch2 = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        gram = kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Shape should be (B1, B2)
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

    def test_gram_matrix_symmetry_real_data(self, test_batches, num_event_types):
        """Test that Gram matrix is symmetric with same batch."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        gram = kernel.graam_matrix(batch1, batch1)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Matrix should be symmetric
        assert torch.allclose(gram, gram.t(), atol=1e-4)

    def test_gram_matrix_positive_diagonal_real_data(
        self, test_batches, num_event_types
    ):
        """Test that diagonal elements are positive with real data."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        gram = kernel.graam_matrix(batch1, batch1)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Diagonal should be positive
        assert torch.all(torch.diag(gram) > 0)

    def test_gram_matrix_rbf_kernel_real_data(self, test_batches, num_event_types):
        """Test Gram matrix with RBF kernel and real data."""
        batch1, batch2 = test_batches

        kernel = SIGKernel(
            static_kernel_type="rbf",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
            sigma=1.0,
        )

        gram = kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Should have correct shape
        assert gram.shape == (batch1.time_seqs.shape[0], batch2.time_seqs.shape[0])

        # Self-similarity should be symmetric
        gram_self = kernel.graam_matrix(batch1, batch1)
        assert torch.allclose(gram_self, gram_self.t(), atol=1e-4)


class TestMMDWithSIGKernel:
    """Test MMD metric with SIG kernel on real data."""

    def test_mmd_self_similarity(self, test_batches, num_event_types):
        """Test that MMD(batch, batch) is close to 0."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        mmd = MMD(kernel=kernel)
        mmd_value = mmd(batch1, batch1)

        # Verify no NaN
        assert not torch.isnan(torch.tensor(mmd_value))
        assert not torch.isinf(torch.tensor(mmd_value))

        # Self-similarity should be very small
        assert abs(mmd_value) < 1e-5

    def test_mmd_different_batches(self, test_batches, num_event_types):
        """Test MMD between different batches."""
        batch1, batch2 = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        mmd = MMD(kernel=kernel)
        mmd_value = mmd(batch1, batch2)

        # Verify no NaN
        assert not torch.isnan(torch.tensor(mmd_value))
        assert not torch.isinf(torch.tensor(mmd_value))

        # MMD should be positive or zero
        assert mmd_value >= 0.0

    def test_mmd_with_noise(self, test_batches, num_event_types):
        """Test that MMD increases with noise level."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        mmd = MMD(kernel=kernel)

        # Test different noise levels
        noise_levels = [0.01, 0.05, 0.1]
        mmd_values = []

        for noise in noise_levels:
            # Create noisy batch
            noisy_times = batch1.time_seqs + torch.randn_like(batch1.time_seqs) * noise
            noisy_times = torch.clamp(noisy_times, min=0)

            batch_noisy = Batch(
                time_seqs=noisy_times,
                time_delta_seqs=batch1.time_delta_seqs,
                type_seqs=batch1.type_seqs,
                valid_event_mask=batch1.valid_event_mask,
            )

            mmd_val = mmd(batch1, batch_noisy)
            mmd_values.append(mmd_val)

        # MMD should generally increase with noise
        # (at least the largest noise should have higher MMD than no noise)
        assert mmd_values[-1] > 1e-6

    def test_mmd_constant_interpolant(self, test_batches, num_event_types):
        """Test MMD with constant interpolant."""
        batch1, batch2 = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="constant_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        mmd = MMD(kernel=kernel)

        # Self-similarity
        mmd_self = mmd(batch1, batch1)
        assert abs(mmd_self) < 1e-5

        # Different batches
        mmd_diff = mmd(batch1, batch2)
        assert mmd_diff >= 0.0

    def test_mmd_rbf_vs_linear(self, test_batches, num_event_types):
        """Compare MMD values with RBF vs linear kernel."""
        batch1, batch2 = test_batches

        kernel_linear = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        kernel_rbf = SIGKernel(
            static_kernel_type="rbf",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
            sigma=1.0,
        )

        mmd_linear = MMD(kernel=kernel_linear)
        mmd_rbf = MMD(kernel=kernel_rbf)

        mmd_val_linear = mmd_linear(batch1, batch2)
        mmd_val_rbf = mmd_rbf(batch1, batch2)

        # Both should be non-negative
        assert mmd_val_linear >= 0.0
        assert mmd_val_rbf >= 0.0


class TestDTypePreservation:
    """Test that dtypes are preserved correctly."""

    def test_embedding_dtype_preservation(self, test_batches, num_event_types):
        """Test that embedding preserves input dtype."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        # Get original dtype
        original_dtype = batch1.time_seqs.dtype

        # Get embedding
        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)

        # Verify no NaN or Inf
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()

        # Check dtype is preserved
        assert embedding.dtype == original_dtype

    def test_gram_matrix_dtype_preservation(self, test_batches, num_event_types):
        """Test that Gram matrix preserves dtype."""
        batch1, batch2 = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        original_dtype = batch1.time_seqs.dtype
        gram = kernel.graam_matrix(batch1, batch2)

        # Verify no NaN or Inf
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()

        # Gram matrix should preserve dtype
        assert gram.dtype == original_dtype


class TestNoNaNOrInf:
    """Test that operations don't produce NaN or Inf values."""

    def test_embedding_no_nan_with_real_data(self, test_batches, num_event_types):
        """Test that embeddings don't contain NaN or Inf."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)

        # Check no NaN or Inf
        assert not torch.isnan(embedding).any(), "Embedding contains NaN values"
        assert not torch.isinf(embedding).any(), "Embedding contains Inf values"

    def test_gram_matrix_no_nan(self, test_batches, num_event_types):
        """Test that Gram matrix doesn't contain NaN or Inf."""
        batch1, batch2 = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        gram = kernel.graam_matrix(batch1, batch2)

        # Check no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN values"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf values"

    def test_mmd_no_nan(self, test_batches, num_event_types):
        """Test that MMD computation doesn't produce NaN."""
        batch1, batch2 = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        mmd = MMD(kernel=kernel)
        mmd_value = mmd(batch1, batch2)

        # Check MMD value is valid
        assert not torch.isnan(torch.tensor(mmd_value)), "MMD value is NaN"
        assert not torch.isinf(torch.tensor(mmd_value)), "MMD value is Inf"

    def test_constant_interpolant_no_nan(self, test_batches, num_event_types):
        """Test constant interpolant doesn't produce NaN."""
        batch1, _ = test_batches

        kernel = SIGKernel(
            static_kernel_type="linear",
            embedding_type="constant_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
        )

        embedding = kernel._get_embedding(batch1.time_seqs, batch1.type_seqs)

        # Check no NaN or Inf
        assert not torch.isnan(embedding).any(), "Embedding contains NaN values"
        assert not torch.isinf(embedding).any(), "Embedding contains Inf values"

    def test_rbf_kernel_no_nan(self, test_batches, num_event_types):
        """Test RBF kernel doesn't produce NaN."""
        batch1, batch2 = test_batches

        kernel = SIGKernel(
            static_kernel_type="rbf",
            embedding_type="linear_interpolant",
            dyadic_order=2,
            num_event_types=max(2, num_event_types),
            sigma=1.0,
        )

        gram = kernel.graam_matrix(batch1, batch2)

        # Check no NaN or Inf
        assert not torch.isnan(gram).any(), "Gram matrix contains NaN values"
        assert not torch.isinf(gram).any(), "Gram matrix contains Inf values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
