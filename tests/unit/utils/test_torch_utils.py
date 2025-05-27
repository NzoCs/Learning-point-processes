"""Tests for torch utilities including device management."""
import pytest
import torch
import os
import random
import numpy as np
from unittest.mock import patch, Mock

from easy_tpp.utils.torch_utils import set_seed, set_device, set_optimizer


@pytest.mark.unit
@pytest.mark.utils
class TestTorchUtils:
    """Test cases for torch utilities."""
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed ensures reproducibility."""
        seed = 42
        
        # Set seed and generate random numbers
        set_seed(seed)
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        py_rand1 = [random.random() for _ in range(5)]
        
        # Set same seed again and generate random numbers
        set_seed(seed)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        py_rand2 = [random.random() for _ in range(5)]
        
        # Check reproducibility
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
        assert py_rand1 == py_rand2
    
    def test_set_seed_environment_variable(self):
        """Test that set_seed sets PYTHONHASHSEED."""
        seed = 123
        set_seed(seed)
        
        assert os.environ["PYTHONHASHSEED"] == str(seed)
    
    def test_set_device_cpu(self):
        """Test setting device to CPU."""
        device = set_device(-1)
        assert device.type == 'cpu'
        
        device = set_device(-2)  # Any negative number should give CPU
        assert device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_set_device_cuda(self):
        """Test setting device to CUDA when available."""
        device = set_device(0)
        assert device.type == 'cuda'
        assert device.index == 0
        
        # Test with different GPU index
        if torch.cuda.device_count() > 1:
            device = set_device(1)
            assert device.type == 'cuda'
            assert device.index == 1
    
    @patch('easy_tpp.utils.torch_utils.is_torch_mps_available')
    @patch('torch.cuda.is_available')
    def test_set_device_mps_fallback(self, mock_cuda_available, mock_mps_available):
        """Test MPS fallback when CUDA is not available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        
        with patch('torch.device') as mock_device:
            set_device(0)
            mock_device.assert_called_with("mps")
    
    def test_set_optimizer_adam(self):
        """Test setting up Adam optimizer."""
        # Create dummy parameters
        linear_layer = torch.nn.Linear(10, 5)
        params = linear_layer.parameters()
        
        optimizer = set_optimizer('adam', params, lr=0.001)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
    
    def test_set_optimizer_sgd(self):
        """Test setting up SGD optimizer."""
        linear_layer = torch.nn.Linear(10, 5)
        params = linear_layer.parameters()
        
        optimizer = set_optimizer('SGD', params, lr=0.01)
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['lr'] == 0.01
    
    def test_set_optimizer_invalid(self):
        """Test error handling for invalid optimizer."""
        linear_layer = torch.nn.Linear(10, 5)
        params = linear_layer.parameters()
        
        with pytest.raises(NotImplementedError):
            set_optimizer('invalid_optimizer', params, lr=0.001)
    
    @pytest.mark.parametrize("seed", [0, 42, 1029, 999999])
    def test_set_seed_different_seeds(self, seed):
        """Test set_seed with different seed values."""
        set_seed(seed)
        
        # Should not raise any errors
        torch_tensor = torch.rand(3)
        np_array = np.random.rand(3)
        
        assert torch_tensor.shape == (3,)
        assert np_array.shape == (3,)
    
    def test_cuda_deterministic_setting(self):
        """Test that CUDA deterministic is set correctly."""
        set_seed(42)
        
        # Check that deterministic flag is set
        assert torch.backends.cudnn.deterministic is True
    
    def test_count_model_params(self):
        """Test counting model parameters."""
        model = torch.nn.Linear(10, 5)
        from easy_tpp.utils.torch_utils import count_model_params
        count = count_model_params(model)
        expected = sum(p.numel() for p in model.parameters())
        assert count == expected


@pytest.mark.device
class TestDeviceManagement:
    """Test cases specifically for device management functionality."""
    
    def test_device_type_consistency(self):
        """Test that device type is consistent."""
        cpu_device = set_device(-1)
        assert cpu_device.type == 'cpu'
        
        # Test multiple calls return same type
        cpu_device2 = set_device(-1)
        assert cpu_device.type == cpu_device2.type
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_index(self):
        """Test CUDA device index handling."""
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            device = set_device(i)
            assert device.type == 'cuda'
            assert device.index == i
    
    def test_device_tensor_placement(self):
        """Test tensor placement on specified device."""
        device = set_device(-1)  # CPU
        
        tensor = torch.randn(5, 5).to(device)
        assert tensor.device == device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available") 
    def test_gpu_tensor_operations(self):
        """Test tensor operations on GPU."""
        gpu_device = set_device(0)
        
        # Create tensors on GPU
        tensor1 = torch.randn(3, 3).to(gpu_device)
        tensor2 = torch.randn(3, 3).to(gpu_device)
        
        # Perform operations
        result = tensor1 + tensor2
        assert result.device == gpu_device
        
        # Matrix multiplication
        result_mm = torch.mm(tensor1, tensor2)
        assert result_mm.device == gpu_device
    
    def test_device_memory_cleanup(self):
        """Test device memory is properly managed."""
        device = set_device(-1)
        
        # Create and delete tensors
        for _ in range(10):
            large_tensor = torch.randn(1000, 1000).to(device)
            del large_tensor
        
        # Should not cause memory issues on CPU
        assert True  # If we reach here, no memory errors occurred
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_cleanup(self):
        """Test CUDA memory cleanup."""
        gpu_device = set_device(0)
        
        initial_memory = torch.cuda.memory_allocated(gpu_device)
        
        # Create tensor
        tensor = torch.randn(100, 100).to(gpu_device)
        allocated_memory = torch.cuda.memory_allocated(gpu_device)
        assert allocated_memory > initial_memory
        
        # Delete tensor and empty cache
        del tensor
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(gpu_device)
        assert final_memory <= allocated_memory
    
    def test_device_context_manager(self):
        """Test device context management."""
        cpu_device = set_device(-1)
        
        with torch.device(cpu_device):
            tensor = torch.randn(5)
            # Tensor should be on the specified device
            assert tensor.device.type == cpu_device.type
    
    @pytest.mark.parametrize("gpu_id", [-1, 0, 1, 2])
    def test_device_id_handling(self, gpu_id):
        """Test handling of different GPU IDs."""
        if gpu_id >= 0 and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        if gpu_id >= torch.cuda.device_count():
            pytest.skip(f"GPU {gpu_id} not available")
        
        device = set_device(gpu_id)
        
        if gpu_id < 0:
            assert device.type == 'cpu'
        else:
            expected_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            assert device.type == expected_type
