#!/usr/bin/env python3
"""
Test script to verify TPPDataset integration with TemporalPointProcessComparatorFactory.

This script demonstrates the performance improvement when using TPPDataset directly
instead of DataLoader for data extraction.
"""

import time
import numpy as np
from easy_tpp.preprocess.dataset import TPPDataset
from easy_tpp.evaluate.distribution_analysis_helper import TemporalPointProcessComparatorFactory


def create_sample_data(num_sequences=100, max_seq_length=50):
    """Create sample TPPDataset for testing."""
    np.random.seed(42)
    
    time_seqs = []
    time_delta_seqs = []
    type_seqs = []
    
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
        'time_seqs': time_seqs,
        'time_delta_seqs': time_delta_seqs,
        'type_seqs': type_seqs
    }
    
    return TPPDataset(data)


def create_sample_simulation(num_sequences=50):
    """Create sample simulation data."""
    np.random.seed(123)
    simulation = []
    
    for _ in range(num_sequences):
        seq_length = np.random.randint(5, 30)
        time_deltas = np.random.exponential(0.4, seq_length)
        event_types = np.random.randint(1, 4, seq_length)
        
        simulation.append({
            'time_delta_seq': time_deltas,
            'event_seq': event_types
        })
    
    return simulation


def test_tpp_dataset_integration():
    """Test the TPPDataset integration with the comparator factory."""
    print("=== Testing TPPDataset Integration ===")
    
    # Create sample data
    print("Creating sample data...")
    tpp_dataset = create_sample_data(num_sequences=200)
    simulation = create_sample_simulation(num_sequences=100)
    
    print(f"Created TPPDataset with {len(tpp_dataset)} sequences")
    print(f"Created simulation with {len(simulation)} sequences")
    
    # Test the factory with TPPDataset
    print("\nTesting TemporalPointProcessComparatorFactory with TPPDataset...")
    
    try:
        start_time = time.time()
        
        comparator = TemporalPointProcessComparatorFactory.create_comparator(
            label_data=tpp_dataset,
            simulation=simulation,
            num_event_types=3,
            output_dir="./test_output",
            dataset_size=1000,
            auto_run=False  # Don't run the actual comparison, just test creation
        )
        
        end_time = time.time()
        
        print(f"✅ Successfully created comparator in {end_time - start_time:.4f} seconds")
        print(f"✅ Comparator type: {type(comparator).__name__}")
        print(f"✅ Label extractor type: {type(comparator.label_extractor).__name__}")
        
        # Test data extraction
        print("\nTesting data extraction...")
        time_deltas = comparator.label_extractor.extract_time_deltas()
        event_types = comparator.label_extractor.extract_event_types()
        seq_lengths = comparator.label_extractor.extract_sequence_lengths()
        
        print(f"✅ Extracted {len(time_deltas)} time deltas")
        print(f"✅ Extracted {len(event_types)} event types")
        print(f"✅ Extracted {len(seq_lengths)} sequence lengths")
        print(f"✅ Average sequence length: {np.mean(seq_lengths):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_tpp_dataset_integration()
    if success:
        print("\n🎉 All tests passed! TPPDataset integration is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the implementation.")
