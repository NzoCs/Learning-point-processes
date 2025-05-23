"""Tests for RMTPP (Recurrent Marked Temporal Point Process) model."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from easy_tpp.models.rmtpp import RMTPP
from easy_tpp.config_factory import ModelConfig


@pytest.mark.unit
@pytest.mark.model
class TestRMTPP:
    """Test cases for RMTPP model."""
    
    def test_rmtpp_initialization(self, sample_model_config):
        """Test RMTPP model initialization."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        
        assert model.model_config.model_id == 'RMTPP'
        assert hasattr(model, 'layer_event_emb')
        assert hasattr(model, 'layer_rnn')
        assert hasattr(model, 'layer_intensity')
        
        # Check RNN type
        assert isinstance(model.layer_rnn, (nn.LSTM, nn.GRU, nn.RNN))
    
    def test_rmtpp_forward(self, sample_model_config, sample_batch_data):
        """Test RMTPP forward pass."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_batch_data)
        
        assert isinstance(output, dict)
        assert 'lambda_at_event' in output
        assert 'hidden_states' in output
        
        batch_size, seq_len = sample_batch_data['time_seqs'].shape
        assert output['lambda_at_event'].shape == (batch_size, seq_len)
        assert output['hidden_states'].shape == (batch_size, seq_len, sample_model_config.hidden_size)
    
    def test_rmtpp_rnn_layers(self, sample_model_config):
        """Test RMTPP RNN layers configuration."""
        sample_model_config.model_id = 'RMTPP'
        
        # Test different RNN configurations
        rnn_configs = [
            {'rnn_type': 'LSTM', 'num_layers': 1},
            {'rnn_type': 'GRU', 'num_layers': 2},
            {'rnn_type': 'RNN', 'num_layers': 1}
        ]
        
        for rnn_config in rnn_configs:
            # Update config
            for key, value in rnn_config.items():
                if hasattr(sample_model_config, key):
                    setattr(sample_model_config, key, value)
            
            try:
                model = RMTPP(sample_model_config)
                assert model is not None
                
                # Check RNN properties
                if hasattr(model.layer_rnn, 'num_layers'):
                    expected_layers = rnn_config.get('num_layers', 1)
                    assert model.layer_rnn.num_layers == expected_layers
                    
            except (AttributeError, ValueError):
                # Config might not support this parameter
                pass
    
    def test_rmtpp_intensity_computation(self, sample_model_config, sample_batch_data):
        """Test intensity computation in RMTPP."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_batch_data)
        
        # Intensity should be positive
        assert torch.all(output['lambda_at_event'] >= 0)
        
        # Check intensity values are reasonable (not NaN or inf)
        assert torch.all(torch.isfinite(output['lambda_at_event']))
    
    def test_rmtpp_hidden_state_propagation(self, sample_model_config):
        """Test hidden state propagation through RNN."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        
        batch_size = 2
        seq_len = 5
        
        # Create sequential input
        time_seqs = torch.rand(batch_size, seq_len)
        type_seqs = torch.randint(1, sample_model_config.num_event_types + 1, (batch_size, seq_len))
        
        batch_data = {
            'time_seqs': time_seqs,
            'type_seqs': type_seqs,
            'seq_lens': torch.full((batch_size,), seq_len),
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'batch_non_pad_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'type_mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
        }
        
        model.eval()
        with torch.no_grad():
            output = model(batch_data)
        
        # Hidden states should have temporal dependencies
        hidden_states = output['hidden_states']
        
        # Check that hidden states change across time steps
        if seq_len > 1:
            # Hidden states at different time steps should be different
            assert not torch.allclose(hidden_states[:, 0, :], hidden_states[:, 1, :])
    
    def test_rmtpp_compute_loglikelihood(self, sample_model_config, sample_batch_data):
        """Test log-likelihood computation."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        
        loglik = model.compute_loglikelihood(sample_batch_data)
        
        assert isinstance(loglik, torch.Tensor)
        assert loglik.shape == (sample_batch_data['time_seqs'].shape[0],)  # Batch size
        
        # Log-likelihood should be finite
        assert torch.all(torch.isfinite(loglik))
    
    def test_rmtpp_embedding_layer(self, sample_model_config):
        """Test event embedding layer in RMTPP."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        
        # Test embedding
        event_types = torch.randint(0, sample_model_config.num_event_types_pad, (4, 10))
        embeddings = model.layer_event_emb(event_types)
        
        assert embeddings.shape == (4, 10, sample_model_config.hidden_size)
        
        # Different event types should have different embeddings
        if sample_model_config.num_event_types > 1:
            emb_type_0 = model.layer_event_emb(torch.tensor([[0]]))
            emb_type_1 = model.layer_event_emb(torch.tensor([[1]]))
            assert not torch.allclose(emb_type_0, emb_type_1)
    
    def test_rmtpp_gradient_flow(self, sample_model_config, sample_batch_data):
        """Test gradient flow through RMTPP model."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        model.train()
        
        # Forward pass
        loss = model.training_step(sample_batch_data, batch_idx=0)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are non-zero
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        assert len(grad_norms) > 0, "No gradients found"
        assert any(norm > 0 for norm in grad_norms), "All gradients are zero"
    
    @pytest.mark.parametrize("hidden_size", [16, 32, 64, 128])
    def test_rmtpp_different_hidden_sizes(self, sample_model_config, hidden_size):
        """Test RMTPP with different hidden sizes."""
        sample_model_config.model_id = 'RMTPP'
        sample_model_config.hidden_size = hidden_size
        
        model = RMTPP(sample_model_config)
        
        # Check embedding and RNN dimensions
        assert model.layer_event_emb.embedding_dim == hidden_size
        assert model.layer_rnn.hidden_size == hidden_size
    
    def test_rmtpp_batch_independence(self, sample_model_config):
        """Test that batch elements are processed independently."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        model.eval()
        
        # Create batch with identical sequences
        seq_len = 5
        time_seq = torch.rand(1, seq_len)
        type_seq = torch.randint(1, sample_model_config.num_event_types + 1, (1, seq_len))
        
        # Single sequence
        single_batch = {
            'time_seqs': time_seq,
            'type_seqs': type_seq,
            'seq_lens': torch.tensor([seq_len]),
            'attention_mask': torch.ones(1, seq_len, dtype=torch.bool),
            'batch_non_pad_mask': torch.ones(1, seq_len, dtype=torch.bool),
            'type_mask': torch.ones(1, seq_len, dtype=torch.bool)
        }
        
        # Batch with two identical sequences
        double_batch = {
            'time_seqs': time_seq.repeat(2, 1),
            'type_seqs': type_seq.repeat(2, 1),
            'seq_lens': torch.tensor([seq_len, seq_len]),
            'attention_mask': torch.ones(2, seq_len, dtype=torch.bool),
            'batch_non_pad_mask': torch.ones(2, seq_len, dtype=torch.bool),
            'type_mask': torch.ones(2, seq_len, dtype=torch.bool)
        }
        
        with torch.no_grad():
            single_output = model(single_batch)
            double_output = model(double_batch)
        
        # Both sequences in double batch should produce same output as single
        assert torch.allclose(
            single_output['lambda_at_event'][0], 
            double_output['lambda_at_event'][0], 
            rtol=1e-5
        )
        assert torch.allclose(
            single_output['lambda_at_event'][0], 
            double_output['lambda_at_event'][1], 
            rtol=1e-5
        )
    
    def test_rmtpp_device_consistency(self, sample_model_config, device):
        """Test device consistency for RMTPP model."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        model = model.to(device)
        
        # Check all parameters are on correct device
        for param in model.parameters():
            assert param.device == device
        
        # Test forward pass on device
        batch_size = 2
        seq_len = 5
        
        batch_data = {
            'time_seqs': torch.rand(batch_size, seq_len).to(device),
            'type_seqs': torch.randint(1, sample_model_config.num_event_types + 1, (batch_size, seq_len)).to(device),
            'seq_lens': torch.full((batch_size,), seq_len).to(device),
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool).to(device),
            'batch_non_pad_mask': torch.ones(batch_size, seq_len, dtype=torch.bool).to(device),
            'type_mask': torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
        }
        
        with torch.no_grad():
            output = model(batch_data)
        
        # Check output tensors are on correct device
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                assert value.device == device
    
    def test_rmtpp_state_dict_consistency(self, sample_model_config):
        """Test state dict save/load consistency."""
        sample_model_config.model_id = 'RMTPP'
        model1 = RMTPP(sample_model_config)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state dict
        model2 = RMTPP(sample_model_config)
        model2.load_state_dict(state_dict)
        
        # Models should have identical parameters
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2
            assert torch.equal(param1, param2)
    
    def test_rmtpp_training_validation_modes(self, sample_model_config, sample_batch_data):
        """Test RMTPP in training and validation modes."""
        sample_model_config.model_id = 'RMTPP'
        model = RMTPP(sample_model_config)
        
        # Training mode
        model.train()
        train_loss = model.training_step(sample_batch_data, batch_idx=0)
        assert train_loss.requires_grad
        
        # Validation mode
        model.eval()
        val_loss = model.validation_step(sample_batch_data, batch_idx=0)
        assert not val_loss.requires_grad
        
        # Test mode
        test_result = model.test_step(sample_batch_data, batch_idx=0)
        assert isinstance(test_result, dict)
        assert 'test_loss' in test_result
