#!/usr/bin/env python3
"""
Integration tests for TPPDataset with NTPPComparatorFactory.

This module tests the performance improvement when using TPPDataset directly
instead of DataLoader for data extraction in the comparator factory.
"""

import pytest
import time
import numpy as np
import tempfile
import shutil
from easy_tpp.data_preprocess.dataset import TPPDataset
from easy_tpp.evaluation.distribution_analysis_helper import (
    NTPPComparatorFactory,
)


class TestTPPDatasetComparatorIntegration:
    """Test suite for TPPDataset integration with comparator factory."""

    @pytest.fixture
    def sample_tpp_dataset(self):
        """Create sample TPPDataset for testing."""
        np.random.seed(42)

        time_seqs = []
        time_delta_seqs = []
        type_seqs = []

        num_sequences = 50
        max_seq_length = 30

        for _ in range(num_sequences):
            seq_length = np.random.randint(5, max_seq_length)

            # Generate time deltas
            time_deltas = np.random.exponential(0.5, seq_length)
            time_delta_seqs.append(time_deltas.tolist())

            # Generate cumulative times
            times = np.cumsum(time_deltas)
            time_seqs.append(times.tolist())

            # Generate event types
            types = np.random.randint(1, 4, seq_length)
            type_seqs.append(types.tolist())

        data = {
            "time_seqs": time_seqs,
            "time_delta_seqs": time_delta_seqs,
            "type_seqs": type_seqs,
        }

        return TPPDataset(data)

    @pytest.fixture
    def sample_simulation(self):
        """Create sample simulation data."""
        np.random.seed(123)
        simulation = []

        for _ in range(25):
            seq_length = np.random.randint(5, 20)
            time_deltas = np.random.exponential(0.4, seq_length)
            event_types = np.random.randint(1, 4, seq_length)

            simulation.append({"time_delta_seq": time_deltas, "event_seq": event_types})

        return simulation

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_tpp_dataset_factory_creation(
        self, sample_tpp_dataset, sample_simulation, temp_output_dir
    ):
        """Test that the factory correctly creates comparator with TPPDataset."""
        comparator = NTPPComparatorFactory.create_comparator(
            label_data=sample_tpp_dataset,
            simulation=sample_simulation,
            num_event_types=3,
            output_dir=temp_output_dir,
            dataset_size=500,
            auto_run=False,
        )

        # Verify the comparator was created with correct extractor type
        assert comparator is not None
        assert hasattr(comparator, "label_extractor")
        assert hasattr(comparator, "simulation_extractor")

        # Check that the correct extractor type is used
        from easy_tpp.evaluation.distribution_analysis_helper.data_extractors import (
            TPPDatasetExtractor,
        )

        assert isinstance(comparator.label_extractor, TPPDatasetExtractor)

    def test_tpp_dataset_data_extraction(
        self, sample_tpp_dataset, sample_simulation, temp_output_dir
    ):
        """Test data extraction functionality with TPPDataset."""
        comparator = NTPPComparatorFactory.create_comparator(
            label_data=sample_tpp_dataset,
            simulation=sample_simulation,
            num_event_types=3,
            output_dir=temp_output_dir,
            dataset_size=300,
            auto_run=False,
        )

        # Extract data
        time_deltas = comparator.label_extractor.extract_time_deltas()
        event_types = comparator.label_extractor.extract_event_types()
        seq_lengths = comparator.label_extractor.extract_sequence_lengths()

        # Verify extracted data
        assert len(time_deltas) > 0
        assert len(event_types) > 0
        assert len(seq_lengths) > 0
        assert len(time_deltas) == len(event_types)

        # Check data types
        assert isinstance(time_deltas, np.ndarray)
        assert isinstance(event_types, np.ndarray)
        assert isinstance(seq_lengths, list)

        # Check data ranges
        assert np.all(time_deltas > 0)  # Time deltas should be positive
        assert np.all(event_types >= 1) and np.all(
            event_types <= 3
        )  # Event types in range

    def test_performance_improvement_indication(
        self, sample_tpp_dataset, sample_simulation, temp_output_dir
    ):
        """Test that TPPDataset extraction performs as expected."""
        start_time = time.time()

        comparator = NTPPComparatorFactory.create_comparator(
            label_data=sample_tpp_dataset,
            simulation=sample_simulation,
            num_event_types=3,
            output_dir=temp_output_dir,
            dataset_size=500,
            auto_run=False,
        )

        # Extract data
        _ = comparator.label_extractor.extract_time_deltas()
        _ = comparator.label_extractor.extract_event_types()
        _ = comparator.label_extractor.extract_sequence_lengths()

        end_time = time.time()
        extraction_time = end_time - start_time

        # Should complete reasonably quickly (less than 1 second for small dataset)
        assert (
            extraction_time < 1.0
        ), f"Extraction took too long: {extraction_time:.4f} seconds"

    def test_backward_compatibility_with_non_tpp_dataset(
        self, sample_simulation, temp_output_dir
    ):
        """Test that factory still works with non-TPPDataset objects (backward compatibility)."""

        # Create a mock DataLoader-like object
        class MockDataLoader:
            def __init__(self):
                self.data = []

        mock_dataloader = MockDataLoader()

        # This should not raise an error, but fall back to LabelDataExtractor
        comparator = NTPPComparatorFactory.create_comparator(
            label_data=mock_dataloader,
            simulation=sample_simulation,
            num_event_types=3,
            output_dir=temp_output_dir,
            dataset_size=100,
            auto_run=False,
        )

        # Verify the fallback extractor is used
        from easy_tpp.evaluation.distribution_analysis_helper.data_extractors import (
            LabelDataExtractor,
        )

        assert isinstance(comparator.label_extractor, LabelDataExtractor)


def test_integration_standalone():
    """Standalone integration test that can be run manually."""
    print("=== Running TPPDataset Integration Test ===")

    # Create sample data
    np.random.seed(42)
    time_seqs = []
    time_delta_seqs = []
    type_seqs = []

    for _ in range(20):
        seq_length = np.random.randint(5, 15)
        time_deltas = np.random.exponential(0.5, seq_length)
        time_delta_seqs.append(time_deltas.tolist())
        times = np.cumsum(time_deltas)
        time_seqs.append(times.tolist())
        types = np.random.randint(1, 3, seq_length)
        type_seqs.append(types.tolist())

    data = {
        "time_seqs": time_seqs,
        "time_delta_seqs": time_delta_seqs,
        "type_seqs": type_seqs,
    }

    tpp_dataset = TPPDataset(data)

    # Create simulation
    simulation = []
    for _ in range(10):
        seq_length = np.random.randint(5, 10)
        simulation.append(
            {
                "time_delta_seq": np.random.exponential(0.4, seq_length),
                "event_seq": np.random.randint(1, 3, seq_length),
            }
        )

    # Test factory
    with tempfile.TemporaryDirectory() as temp_dir:
        comparator = NTPPComparatorFactory.create_comparator(
            label_data=tpp_dataset,
            simulation=simulation,
            num_event_types=2,
            output_dir=temp_dir,
            dataset_size=200,
            auto_run=False,
        )

        print(f"✅ Comparator created successfully")
        print(f"✅ Label extractor type: {type(comparator.label_extractor).__name__}")

        # Test extraction
        time_deltas = comparator.label_extractor.extract_time_deltas()
        print(f"✅ Extracted {len(time_deltas)} time deltas")

    print("🎉 Integration test completed successfully!")


if __name__ == "__main__":
    test_integration_standalone()
