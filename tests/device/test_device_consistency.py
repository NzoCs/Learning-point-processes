"""Device consistency tests for GPU/CPU handling during model training and execution."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import pytorch_lightning as pl

from easy_tpp.models.nhp import NHP
from easy_tpp.models.rmtpp import RMTPP
from easy_tpp.utils.torch_utils import set_device
from easy_tpp.config_factory import ModelConfig
from tests.conftest import check_device_consistency, check_tensor_device


@pytest.mark.device
class TestDeviceConsistency:
    """Test device consistency across different scenarios."""
    
    def test_model_parameter_device_consistency(self, sample_model_config):
        """Test that all model parameters are on the same device."""
        model = NHP(sample_model_config)
        
        # Test CPU placement
        cpu_device = torch.device('cpu')
        model = model.to(cpu_device)
        check_device_consistency(model, cpu_device)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_gpu_device_consistency(self, sample_model_config):
        """Test model device consistency on GPU."""
        model = NHP(sample_model_config)
        
        # Test GPU placement
        gpu_device = torch.device('cuda:0')
        model = model.to(gpu_device)
        check_device_consistency(model, gpu_device)
    
    def test_batch_data_device_consistency(self, sample_batch_data):
        """Test that batch data can be moved to different devices consistently."""
        cpu_device = torch.device('cpu')
        # Move all tensors in the tuple to CPU and check
        for tensor in sample_batch_data:
            if isinstance(tensor, torch.Tensor):
                tensor_on_cpu = tensor.to(cpu_device)
                check_tensor_device(tensor_on_cpu, cpu_device)    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_data_gpu_consistency(self, sample_batch_data):
        """Test batch data GPU device consistency."""
        gpu_device = torch.device('cuda:0')
        # Move all tensors in the tuple to GPU and check
        for tensor in sample_batch_data:
            if isinstance(tensor, torch.Tensor):
                tensor_on_gpu = tensor.to(gpu_device)
                check_tensor_device(tensor_on_gpu, gpu_device)
    
    def test_model_forward_device_consistency(self, sample_model_config, sample_batch_data):
        """Test device consistency during training_step."""
        model = NHP(sample_model_config)
        cpu_device = torch.device('cpu')
        model = model.to(cpu_device)
        # Move all tensors in the batch dict to CPU
        cpu_batch = {k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v 
                    for k, v in sample_batch_data.items()}
        model.train()
        # Use training_step instead of forward for proper workflow
        loss = model.training_step(cpu_batch, batch_idx=0)
        # Check loss device consistency
        check_tensor_device(loss, cpu_device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_forward_gpu_consistency(self, sample_model_config, sample_batch_data):
        """Test GPU device consistency during forward pass."""        
        model = NHP(sample_model_config)
        gpu_device = torch.device('cuda:0')
        model = model.to(gpu_device)
        # Move all tensors in the batch dict to GPU
        gpu_batch = {k: v.to(gpu_device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample_batch_data.items()}
        model.train()
        # Use training_step instead of forward for proper workflow
        loss = model.training_step(gpu_batch, batch_idx=0)
        # Check loss device consistency
        check_tensor_device(loss, gpu_device)

    def test_device_switching_during_training(self, sample_model_config, sample_batch_data):
        """Test device switching during training steps."""
        model = NHP(sample_model_config)
        cpu_device = torch.device('cpu')
        model = model.to(cpu_device)
          # Move batch data to CPU
        cpu_batch_dict = {k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v 
                         for k, v in sample_batch_data.items()}
        
        model.train()
        loss_cpu = model.training_step(cpu_batch_dict, batch_idx=0)
        check_tensor_device(loss_cpu, cpu_device)
        if torch.cuda.is_available():
            gpu_device = torch.device('cuda:0')
            model = model.to(gpu_device)
            gpu_batch_dict = {k: v.to(gpu_device) if isinstance(v, torch.Tensor) else v 
                             for k, v in sample_batch_data.items()}
            loss_gpu = model.training_step(gpu_batch_dict, batch_idx=0)
            check_tensor_device(loss_gpu, gpu_device)

    def test_gradient_device_consistency(self, sample_model_config, sample_batch_data):
        """Test that gradients are on the same device as parameters."""
        model = NHP(sample_model_config)
        cpu_device = torch.device('cpu')
        model = model.to(cpu_device)
        cpu_batch_dict = {k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v 
                         for k, v in sample_batch_data.items()}
        model.train()
        loss = model.training_step(cpu_batch_dict, batch_idx=0)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                assert param.device == param.grad.device
                check_tensor_device(param.grad, cpu_device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_gradient_device_consistency(self, sample_model_config, sample_batch_data):
        """Test GPU gradient device consistency."""
        model = NHP(sample_model_config)
        gpu_device = torch.device('cuda:0')
        model = model.to(gpu_device)
        gpu_batch = {k: v.to(gpu_device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample_batch_data.items()}
        model.train()
        loss = model.training_step(gpu_batch, batch_idx=0)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                assert param.device == param.grad.device
                check_tensor_device(param.grad, gpu_device)
    
    def test_optimizer_state_device_consistency(self, sample_model_config):
        """Test that optimizer states are on correct device."""
        model = NHP(sample_model_config)
        cpu_device = torch.device('cpu')
        model = model.to(cpu_device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy gradient
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        # Optimizer step
        optimizer.step()
        
        # Check optimizer state device (if any state exists)
        for group in optimizer.param_groups:
            for param in group['params']:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            check_tensor_device(value, cpu_device)
    
    def test_mixed_precision_device_consistency(self, sample_model_config, sample_batch_data):
        """Test device consistency with mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision")
        
        model = NHP(sample_model_config)
        gpu_device = torch.device('cuda:0')
        model = model.to(gpu_device)
        
        gpu_batch = {}
        for key, value in sample_batch_data.items():
            if isinstance(value, torch.Tensor):
                gpu_batch[key] = value.to(gpu_device)
            else:
                gpu_batch[key] = value
        
        # Use autocast for mixed precision
        model.train()
        with torch.autocast(device_type='cuda'):
            output = model(gpu_batch)
            
            # Check output devices
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    # Output might be in different dtype but same device
                    assert value.device.type == gpu_device.type
    
    @pytest.mark.parametrize("model_class", [NHP])  # Add more models as needed
    def test_multiple_model_device_consistency(self, sample_model_config, model_class):
        """Test device consistency across different model types."""
        model = model_class(sample_model_config)
        cpu_device = torch.device('cpu')
        
        model = model.to(cpu_device)
        check_device_consistency(model, cpu_device)
    
    def test_dataloader_device_consistency(self, sample_batch_data):
        """Test device consistency with data loaders."""
        cpu_device = torch.device('cpu')
        # Simulate a dataloader with a list of batches (tuples)
        mock_dataloader = [sample_batch_data]
        for batch in mock_dataloader:
            # Move all tensors in the tuple to CPU and check
            cpu_batch = tuple(tensor.to(cpu_device) if isinstance(tensor, torch.Tensor) else tensor for tensor in batch)
            for tensor in cpu_batch:
                if isinstance(tensor, torch.Tensor):
                    check_tensor_device(tensor, cpu_device)
    
    def test_model_state_dict_device_independence(self, sample_model_config):
        """Test that state dict is device independent."""
        model1 = NHP(sample_model_config)
        cpu_device = torch.device('cpu')
        model1 = model1.to(cpu_device)
        
        # Get state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state dict
        model2 = NHP(sample_model_config)
        model2.load_state_dict(state_dict)
        
        # Models should have same parameters regardless of device
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2
            assert torch.equal(param1.cpu(), param2.cpu())
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_device_state_dict_loading(self, sample_model_config):
        """Test loading state dict across different devices."""
        # Create model on CPU
        model_cpu = NHP(sample_model_config)
        cpu_device = torch.device('cpu')
        model_cpu = model_cpu.to(cpu_device)
        
        # Get state dict
        state_dict = model_cpu.state_dict()
        
        # Create model on GPU and load CPU state dict
        model_gpu = NHP(sample_model_config)
        gpu_device = torch.device('cuda:0')
        model_gpu = model_gpu.to(gpu_device)
        model_gpu.load_state_dict(state_dict)
        
        # Check that GPU model parameters are on GPU
        check_device_consistency(model_gpu, gpu_device)
        
        # Check that values are the same (on CPU for comparison)
        for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(
            model_cpu.named_parameters(), model_gpu.named_parameters()
        ):
            assert name_cpu == name_gpu
            assert torch.equal(param_cpu, param_gpu.cpu())
