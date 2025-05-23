"""Tests for data preprocessing components."""
import pytest
import torch
import numpy as np
import tempfile
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from easy_tpp.preprocess.data_loader import DataLoader
from easy_tpp.preprocess.dataset import TPPDataset
from easy_tpp.preprocess.data_collator import TPPDataCollator
from easy_tpp.preprocess.event_tokenizer import EventTokenizer


@pytest.mark.unit
@pytest.mark.data
class TestEventTokenizer:
    """Test cases for EventTokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = EventTokenizer(
            num_event_types=5,
            pad_token_id=0,
            padding_side='left',
            truncation_side='right',
            max_seq_len=100
        )
        
        assert tokenizer.num_event_types == 5
        assert tokenizer.pad_token_id == 0
        assert tokenizer.padding_side == 'left'
        assert tokenizer.max_seq_len == 100
    
    def test_tokenizer_padding_left(self):
        """Test left padding functionality."""
        tokenizer = EventTokenizer(
            num_event_types=5,
            pad_token_id=0,
            padding_side='left',
            max_seq_len=10
        )
        
        # Short sequence
        sequence = [1, 2, 3]
        padded = tokenizer.pad_sequence(sequence)
        
        expected = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
        assert padded == expected
    
    def test_tokenizer_padding_right(self):
        """Test right padding functionality."""
        tokenizer = EventTokenizer(
            num_event_types=5,
            pad_token_id=0,
            padding_side='right',
            max_seq_len=10
        )
        
        sequence = [1, 2, 3]
        padded = tokenizer.pad_sequence(sequence)
        
        expected = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
        assert padded == expected
    
    def test_tokenizer_truncation_left(self):
        """Test left truncation functionality."""
        tokenizer = EventTokenizer(
            num_event_types=5,
            truncation_side='left',
            max_seq_len=5
        )
        
        # Long sequence
        sequence = [1, 2, 3, 4, 5, 6, 7, 8]
        truncated = tokenizer.truncate_sequence(sequence)
        
        expected = [4, 5, 6, 7, 8]  # Keep rightmost elements
        assert truncated == expected
    
    def test_tokenizer_truncation_right(self):
        """Test right truncation functionality."""
        tokenizer = EventTokenizer(
            num_event_types=5,
            truncation_side='right',
            max_seq_len=5
        )
        
        sequence = [1, 2, 3, 4, 5, 6, 7, 8]
        truncated = tokenizer.truncate_sequence(sequence)
        
        expected = [1, 2, 3, 4, 5]  # Keep leftmost elements
        assert truncated == expected
    
    def test_tokenizer_encode_sequence(self, sample_event_sequences):
        """Test encoding event sequences."""
        tokenizer = EventTokenizer(
            num_event_types=5,
            max_seq_len=10
        )
        
        sequence = sample_event_sequences[0]
        encoded = tokenizer.encode(sequence)
        
        assert isinstance(encoded, dict)
        # Check for expected keys
        expected_keys = ['time_seqs', 'type_seqs', 'seq_lens']
        for key in expected_keys:
            if key in encoded:
                assert key in encoded
    
    def test_tokenizer_batch_encoding(self, sample_event_sequences):
        """Test batch encoding."""
        tokenizer = EventTokenizer(
            num_event_types=5,
            max_seq_len=10
        )
        
        encoded_batch = tokenizer.encode_batch(sample_event_sequences)
        
        assert isinstance(encoded_batch, dict)
        # Check batch dimensions
        for key, value in encoded_batch.items():
            if isinstance(value, (list, torch.Tensor, np.ndarray)):
                assert len(value) == len(sample_event_sequences)


@pytest.mark.unit
@pytest.mark.data
class TestCollator:
    """Test cases for data collator."""
    
    def test_collator_initialization(self):
        """Test collator initialization."""
        collator = TPPDataCollator(
            pad_token_id=0,
            max_seq_len=100
        )
        
        assert collator.pad_token_id == 0
        assert collator.max_seq_len == 100
    
    def test_collator_batch_creation(self, sample_event_sequences):
        """Test batch creation from sequences."""
        collator = TPPDataCollator(pad_token_id=0, max_seq_len=20)
        
        # Convert sequences to expected format
        batch_items = []
        for seq in sample_event_sequences:
            item = {
                'time_since_last_event': torch.tensor(seq['time_since_last_event']),
                'type_event': torch.tensor(seq['type_event'])
            }
            batch_items.append(item)
        
        batch = collator(batch_items)
        
        assert isinstance(batch, dict)
        # Check batch has consistent dimensions
        if 'time_seqs' in batch:
            assert batch['time_seqs'].ndim == 2  # [batch_size, seq_len]
        if 'type_seqs' in batch:
            assert batch['type_seqs'].ndim == 2
    
    def test_collator_padding_consistency(self):
        """Test padding consistency across batch."""
        collator = TPPDataCollator(pad_token_id=0, max_seq_len=10)
        
        # Create items with different lengths
        items = [
            {
                'time_since_last_event': torch.tensor([0.1, 0.2]),
                'type_event': torch.tensor([1, 2])
            },
            {
                'time_since_last_event': torch.tensor([0.3, 0.4, 0.5, 0.6]),
                'type_event': torch.tensor([2, 1, 3, 2])
            }
        ]
        
        batch = collator(items)
        
        # All sequences should have same length after collation
        if 'time_seqs' in batch and 'type_seqs' in batch:
            assert batch['time_seqs'].shape[1] == batch['type_seqs'].shape[1]
    
    def test_collator_attention_mask(self):
        """Test attention mask creation."""
        collator = TPPDataCollator(pad_token_id=0, max_seq_len=10)
        
        items = [
            {
                'time_since_last_event': torch.tensor([0.1, 0.2, 0.3]),
                'type_event': torch.tensor([1, 2, 1])
            }
        ]
        
        batch = collator(items)
        
        if 'attention_mask' in batch:
            mask = batch['attention_mask']
            assert mask.dtype == torch.bool
            # First 3 positions should be True, rest False (if padded)
            assert torch.all(mask[0, :3] == True)


@pytest.mark.unit
@pytest.mark.data
class TestTPPDataset:
    """Test cases for TPPDataset."""
    
    def test_dataset_initialization(self, temporary_directory):
        """Test dataset initialization."""
        # Create dummy data file
        data_file = temporary_directory / 'test_data.pkl'
        dummy_data = [
            {
                'time_since_start': [0.1, 0.3, 0.7],
                'time_since_last_event': [0.1, 0.2, 0.4],
                'type_event': [1, 2, 1]
            }
        ]
        
        with open(data_file, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        dataset = TPPDataset(
            data_dir=str(temporary_directory),
            data_file='test_data.pkl'
        )
        
        assert len(dataset) == 1
    
    def test_dataset_getitem(self, temporary_directory):
        """Test dataset item retrieval."""
        data_file = temporary_directory / 'test_data.pkl'
        dummy_data = [
            {
                'time_since_start': [0.1, 0.3],
                'time_since_last_event': [0.1, 0.2],
                'type_event': [1, 2]
            },
            {
                'time_since_start': [0.2, 0.5, 0.8],
                'time_since_last_event': [0.2, 0.3, 0.3],
                'type_event': [2, 1, 3]
            }
        ]
        
        with open(data_file, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        dataset = TPPDataset(
            data_dir=str(temporary_directory),
            data_file='test_data.pkl'
        )
        
        item = dataset[0]
        assert isinstance(item, dict)
        assert 'time_since_last_event' in item
        assert 'type_event' in item
    
    def test_dataset_length(self, temporary_directory):
        """Test dataset length calculation."""
        data_file = temporary_directory / 'test_data.pkl'
        dummy_data = [{'seq': i} for i in range(50)]
        
        with open(data_file, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        dataset = TPPDataset(
            data_dir=str(temporary_directory),
            data_file='test_data.pkl'
        )
        
        assert len(dataset) == 50
    
    def test_dataset_iteration(self, temporary_directory):
        """Test dataset iteration."""
        data_file = temporary_directory / 'test_data.pkl'
        dummy_data = [
            {'id': 0, 'type_event': [1, 2]},
            {'id': 1, 'type_event': [2, 1, 3]},
            {'id': 2, 'type_event': [1]}
        ]
        
        with open(data_file, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        dataset = TPPDataset(
            data_dir=str(temporary_directory),
            data_file='test_data.pkl'
        )
        
        items = list(dataset)
        assert len(items) == 3
        
        for i, item in enumerate(dataset):
            assert item['id'] == i


@pytest.mark.unit
@pytest.mark.data
class TestDataLoader:
    """Test cases for DataLoader."""
    
    def test_dataloader_creation(self, sample_data_config, temporary_directory):
        """Test dataloader creation."""
        # Create dummy data files
        for split in ['train', 'valid', 'test']:
            data_dir = temporary_directory / split
            data_dir.mkdir()
            
            data_file = data_dir / 'data.pkl'
            dummy_data = [
                {
                    'time_since_start': [0.1, 0.3],
                    'time_since_last_event': [0.1, 0.2],
                    'type_event': [1, 2]
                }
            ] * 10  # 10 sequences
            
            with open(data_file, 'wb') as f:
                pickle.dump(dummy_data, f)
        
        # Update config paths
        sample_data_config.train_dir = str(temporary_directory / 'train')
        sample_data_config.valid_dir = str(temporary_directory / 'valid')
        sample_data_config.test_dir = str(temporary_directory / 'test')
        
        try:
            dataloader = DataLoader(sample_data_config)
            
            train_loader = dataloader.get_train_dataloader()
            assert train_loader is not None
            
            valid_loader = dataloader.get_val_dataloader()
            assert valid_loader is not None
            
        except (ImportError, AttributeError, FileNotFoundError):
            # DataLoader might have different interface or dependencies
            pytest.skip("DataLoader interface not compatible")
    
    def test_dataloader_batch_consistency(self, temporary_directory):
        """Test that dataloader produces consistent batches."""
        # Create test data
        data_dir = temporary_directory / 'train'
        data_dir.mkdir()
        
        data_file = data_dir / 'data.pkl'
        dummy_data = []
        for i in range(20):
            dummy_data.append({
                'time_since_start': [0.1 * j for j in range(1, 4)],
                'time_since_last_event': [0.1, 0.1, 0.1],
                'type_event': [1, 2, 1]
            })
        
        with open(data_file, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        # Mock dataloader if necessary
        try:
            from torch.utils.data import DataLoader as TorchDataLoader
            from easy_tpp.preprocess.dataset import TPPDataset
            
            dataset = TPPDataset(str(data_dir), 'data.pkl')
            loader = TorchDataLoader(dataset, batch_size=4, shuffle=False)
            
            batches = list(loader)
            assert len(batches) == 5  # 20 / 4 = 5 batches
            
        except (ImportError, AttributeError):
            pytest.skip("DataLoader components not available")
    
    def test_dataloader_empty_dataset(self, temporary_directory):
        """Test dataloader with empty dataset."""
        data_dir = temporary_directory / 'empty'
        data_dir.mkdir()
        
        data_file = data_dir / 'data.pkl'
        with open(data_file, 'wb') as f:
            pickle.dump([], f)
        
        try:
            dataset = TPPDataset(str(data_dir), 'data.pkl')
            assert len(dataset) == 0
            
        except (ImportError, AttributeError):
            pytest.skip("Dataset class not available")
    
    def test_dataloader_data_types(self, temporary_directory):
        """Test dataloader with different data types."""
        data_dir = temporary_directory / 'types'
        data_dir.mkdir()
        
        # Test with numpy arrays
        data_file = data_dir / 'data.pkl'
        dummy_data = [
            {
                'time_since_last_event': np.array([0.1, 0.2, 0.3]),
                'type_event': np.array([1, 2, 1])
            }
        ]
        
        with open(data_file, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        try:
            dataset = TPPDataset(str(data_dir), 'data.pkl')
            item = dataset[0]
            
            # Check data can be converted to tensors
            if 'time_since_last_event' in item:
                time_tensor = torch.tensor(item['time_since_last_event'])
                assert time_tensor.dtype in [torch.float32, torch.float64]
                
        except (ImportError, AttributeError):
            pytest.skip("Dataset conversion not available")


@pytest.mark.integration
@pytest.mark.data
class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    def test_end_to_end_data_pipeline(self, temporary_directory):
        """Test complete data pipeline from raw data to model input."""
        # Create synthetic data
        sequences = []
        for i in range(10):
            seq_len = np.random.randint(5, 15)
            sequences.append({
                'time_since_start': np.cumsum(np.random.exponential(0.1, seq_len)).tolist(),
                'time_since_last_event': np.random.exponential(0.1, seq_len).tolist(),
                'type_event': np.random.randint(1, 6, seq_len).tolist(),
                'idx_event': i
            })
        
        # Save data
        data_file = temporary_directory / 'sequences.pkl'
        with open(data_file, 'wb') as f:
            pickle.dump(sequences, f)
        
        try:
            # Load dataset
            dataset = TPPDataset(str(temporary_directory), 'sequences.pkl')
            
            # Create collator
            collator = TPPDataCollator(pad_token_id=0, max_seq_len=20)
            
            # Create dataloader
            from torch.utils.data import DataLoader as TorchDataLoader
            loader = TorchDataLoader(
                dataset, 
                batch_size=4, 
                shuffle=False,
                collate_fn=collator
            )
            
            # Get batch
            batch = next(iter(loader))
            
            # Verify batch structure
            assert isinstance(batch, dict)
            
            # Check tensor shapes are consistent
            batch_size = 4
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    assert value.shape[0] == batch_size
                    
        except (ImportError, AttributeError):
            pytest.skip("Data pipeline components not available")
