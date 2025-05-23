"""Tests for NHP (Neural Hawkes Process) model."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from easy_tpp.models.nhp import NHP, ContTimeLSTMCell
from easy_tpp.config_factory import ModelConfig


@pytest.mark.unit
@pytest.mark.model
class TestContTimeLSTMCell:
    """Test cases for ContTimeLSTMCell."""
    
    def test_cell_initialization(self):
        """Test LSTM cell initialization."""
        hidden_dim = 32
        cell = ContTimeLSTMCell(hidden_dim)
        
        assert cell.hidden_dim == hidden_dim
        assert hasattr(cell, 'linear_layer')
        assert hasattr(cell, 'softplus')
        
        # Check linear layer dimensions
        assert cell.linear_layer.in_features == 2 * hidden_dim
        assert cell.linear_layer.out_features == 7 * hidden_dim
    
    def test_cell_forward(self):
        """Test LSTM cell forward pass."""
        hidden_dim = 32
        batch_size = 4
        
        cell = ContTimeLSTMCell(hidden_dim)
        
        # Create inputs
        x_i = torch.randn(batch_size, hidden_dim)
        hidden_ti_minus = torch.randn(batch_size, hidden_dim)
        ct_ti_minus = torch.randn(batch_size, hidden_dim)
        c_bar_im1 = torch.randn(batch_size, hidden_dim)
        
        # Forward pass
        ct_i, c_bar_i, decay_i, output_i = cell(x_i, hidden_ti_minus, ct_ti_minus, c_bar_im1)
        
        # Check output shapes
        assert ct_i.shape == (batch_size, hidden_dim)
        assert c_bar_i.shape == (batch_size, hidden_dim)
        assert decay_i.shape == (batch_size, hidden_dim)
        assert output_i.shape == (batch_size, hidden_dim)
    
    def test_cell_gates_range(self):
        """Test that gates are in correct range."""
        hidden_dim = 16
        batch_size = 2
        
        cell = ContTimeLSTMCell(hidden_dim)
        
        x_i = torch.randn(batch_size, hidden_dim)
        hidden_ti_minus = torch.randn(batch_size, hidden_dim)
        ct_ti_minus = torch.randn(batch_size, hidden_dim)
        c_bar_im1 = torch.randn(batch_size, hidden_dim)
        
        ct_i, c_bar_i, decay_i, output_i = cell(x_i, hidden_ti_minus, ct_ti_minus, c_bar_im1)
        
        # Decay should be positive (softplus output)
        assert torch.all(decay_i >= 0)


@pytest.mark.unit
@pytest.mark.model
class TestNHP:
    """Test cases for Neural Hawkes Process model."""
    
    def test_nhp_initialization(self, sample_model_config):
        """Test NHP model initialization."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        
        assert model.model_config.model_id == 'NHP'
        assert hasattr(model, 'layer_event_emb')
        assert hasattr(model, 'layer_cont_lstm')
        assert hasattr(model, 'layer_intensity')
        assert hasattr(model, 'layer_hidden_output')
        
        # Check embedding dimensions
        assert model.layer_event_emb.num_embeddings == sample_model_config.num_event_types_pad
        assert model.layer_event_emb.embedding_dim == sample_model_config.hidden_size
    
    def test_nhp_forward(self, sample_model_config, sample_batch_data):
        """Test NHP forward pass."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_batch_data)
        
        assert isinstance(output, dict)
        assert 'lambda_at_event' in output
        assert 'hidden_states' in output
        
        batch_size, seq_len = sample_batch_data['time_seqs'].shape
        assert output['lambda_at_event'].shape == (batch_size, seq_len)
        assert output['hidden_states'].shape == (batch_size, seq_len, sample_model_config.hidden_size)
    
    def test_nhp_intensity_computation(self, sample_model_config, sample_batch_data):
        """Test intensity computation in NHP."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_batch_data)
        
        # Intensity should be positive
        assert torch.all(output['lambda_at_event'] >= 0)
    
    def test_nhp_embedding_layer(self, sample_model_config):
        """Test event embedding layer."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        
        # Test embedding
        event_types = torch.randint(0, sample_model_config.num_event_types_pad, (4, 10))
        embeddings = model.layer_event_emb(event_types)
        
        assert embeddings.shape == (4, 10, sample_model_config.hidden_size)
    
    def test_nhp_compute_loglikelihood(self, sample_model_config, sample_batch_data):
        """Test log-likelihood computation."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        
        loglik = model.compute_loglikelihood(sample_batch_data)
        
        assert isinstance(loglik, torch.Tensor)
        assert loglik.shape == (sample_batch_data['time_seqs'].shape[0],)  # Batch size
    
    def test_nhp_state_decay(self, sample_model_config):
        """Test state decay functionality."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        
        batch_size = 4
        hidden_dim = sample_model_config.hidden_size
        
        # Test state decay
        cell_states = torch.randn(batch_size, hidden_dim)
        c_bar = torch.randn(batch_size, hidden_dim)
        decays = torch.rand(batch_size, hidden_dim)  # Positive values
        duration_t = torch.rand(batch_size, 1)  # Time intervals
        
        with torch.no_grad():
            decayed_states = model.state_decay(cell_states, c_bar, decays, duration_t)
        
        assert decayed_states.shape == (batch_size, hidden_dim)
    
    def test_nhp_trainable_parameters(self, sample_model_config):
        """Test that model has trainable parameters."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0
        
        # Check parameter shapes are reasonable
        total_params = sum(p.numel() for p in trainable_params)
        assert total_params > 0
    
    def test_nhp_gradient_flow(self, sample_model_config, sample_batch_data):
        """Test gradient flow through NHP model."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        model.train()
        
        # Forward pass
        loss = model.training_step(sample_batch_data, batch_idx=0)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        grad_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_count += 1
                # Check gradient is not zero everywhere
                assert not torch.all(param.grad == 0)
        
        assert grad_count > 0, "No gradients found"
    
    @pytest.mark.parametrize("hidden_size", [16, 32, 64, 128])
    def test_nhp_different_hidden_sizes(self, sample_model_config, hidden_size):
        """Test NHP with different hidden sizes."""
        sample_model_config.model_id = 'NHP'
        sample_model_config.hidden_size = hidden_size
        
        model = NHP(sample_model_config)
        
        # Check embedding dimension
        assert model.layer_event_emb.embedding_dim == hidden_size
        
        # Check LSTM cell dimension
        assert model.layer_cont_lstm.hidden_dim == hidden_size
    
    def test_nhp_device_consistency(self, sample_model_config, device):
        """Test device consistency for NHP model."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        model = model.to(device)
        
        # Check all parameters are on correct device
        for param in model.parameters():
            assert param.device == device
        
        # Test forward pass on device
        batch_data = {}
        for key, value in sample_batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(device)
            else:
                batch_data[key] = value
        
        with torch.no_grad():
            output = model(batch_data)
        
        # Check output tensors are on correct device
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                assert value.device == device
    
    def test_nhp_eval_mode(self, sample_model_config, sample_batch_data):
        """Test NHP in evaluation mode."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        model.eval()
        
        # Forward pass should work in eval mode
        with torch.no_grad():
            output1 = model(sample_batch_data)
            output2 = model(sample_batch_data)
        
        # Outputs should be identical in eval mode (deterministic)
        assert torch.allclose(output1['lambda_at_event'], output2['lambda_at_event'])
    
    def test_nhp_sequence_lengths(self, sample_model_config):
        """Test NHP with different sequence lengths."""
        sample_model_config.model_id = 'NHP'
        model = NHP(sample_model_config)
        model.eval()
        
        for seq_len in [5, 10, 20, 50]:
            batch_data = {
                'time_seqs': torch.rand(2, seq_len),
                'type_seqs': torch.randint(1, sample_model_config.num_event_types + 1, (2, seq_len)),
                'seq_lens': torch.full((2,), seq_len),
                'attention_mask': torch.ones(2, seq_len, dtype=torch.bool),
                'batch_non_pad_mask': torch.ones(2, seq_len, dtype=torch.bool),
                'type_mask': torch.ones(2, seq_len, dtype=torch.bool)
            }
            
            with torch.no_grad():
                output = model(batch_data)
            
            assert output['lambda_at_event'].shape == (2, seq_len)
