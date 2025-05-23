'''
Unit tests for the THP model.
'''
import pytest
import torch
from easy_tpp.models.thp import THP
from easy_tpp.config_factory import ModelConfig

@pytest.mark.unit
@pytest.mark.model
class TestTHP:
    def test_thp_initialization(self, sample_model_config):
        sample_model_config.model_id = 'THP'
        model = THP(sample_model_config)
        assert model.d_model == sample_model_config.hidden_size
        assert model.n_layers == sample_model_config.num_layers
        assert model.n_head == sample_model_config.num_heads
        assert model.num_event_types == sample_model_config.num_event_types

    def test_thp_forward_pass(self, sample_model_config, sample_batch_data):
        sample_model_config.model_id = 'THP'
        model = THP(sample_model_config)
        model.eval()
        with torch.no_grad():
            enc_output = model.forward(
                sample_batch_data['time_seqs'][:, :-1],
                sample_batch_data['type_seqs'][:, :-1],
                sample_batch_data['attention_mask'][:, :-1, :-1]
            )
        assert enc_output.shape == (
            sample_batch_data['time_seqs'].shape[0],
            sample_batch_data['time_seqs'].shape[1] - 1,
            sample_model_config.hidden_size
        )

    def test_thp_loglike_loss(self, sample_model_config, sample_batch_data):
        sample_model_config.model_id = 'THP'
        model = THP(sample_model_config)
        loss, num_events = model.loglike_loss((
            sample_batch_data['time_seqs'],
            torch.cat((sample_batch_data['time_seqs'][:, :1], sample_batch_data['time_seqs'][:, 1:] - sample_batch_data['time_seqs'][:, :-1]), dim=1),
            sample_batch_data['type_seqs'],
            sample_batch_data['batch_non_pad_mask'],
            sample_batch_data['attention_mask']
        ))
        assert isinstance(loss, torch.Tensor)
        assert isinstance(num_events, torch.Tensor)
        assert loss.dim() == 0
        assert num_events.item() >= 0

    def test_thp_gradient_flow(self, sample_model_config, sample_batch_data):
        sample_model_config.model_id = 'THP'
        model = THP(sample_model_config)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss, _ = model.loglike_loss((
            sample_batch_data['time_seqs'],
            torch.cat((sample_batch_data['time_seqs'][:, :1], sample_batch_data['time_seqs'][:, 1:] - sample_batch_data['time_seqs'][:, :-1]), dim=1),
            sample_batch_data['type_seqs'],
            sample_batch_data['batch_non_pad_mask'],
            sample_batch_data['attention_mask']
        ))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and g.abs().sum().item() > 0 for g in grads)

    def test_thp_state_dict_save_load(self, sample_model_config, sample_batch_data):
        sample_model_config.model_id = 'THP'
        model = THP(sample_model_config)
        model_new = THP(sample_model_config)
        model_new.load_state_dict(model.state_dict())
        model_new.eval()
        model.eval()
        with torch.no_grad():
            enc_output_orig = model.forward(
                sample_batch_data['time_seqs'][:, :-1],
                sample_batch_data['type_seqs'][:, :-1],
                sample_batch_data['attention_mask'][:, :-1, :-1]
            )
            enc_output_new = model_new.forward(
                sample_batch_data['time_seqs'][:, :-1],
                sample_batch_data['type_seqs'][:, :-1],
                sample_batch_data['attention_mask'][:, :-1, :-1]
            )
        assert torch.allclose(enc_output_orig, enc_output_new, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thp_gpu_forward_pass(self, sample_model_config, sample_batch_data):
        sample_model_config.model_id = 'THP'
        gpu_device = torch.device("cuda")
        model_gpu = THP(sample_model_config).to(gpu_device)
        time_seqs_gpu = sample_batch_data['time_seqs'].to(gpu_device)
        type_seqs_gpu = sample_batch_data['type_seqs'].to(gpu_device)
        attention_mask_gpu = sample_batch_data['attention_mask'].to(gpu_device)
        enc_output_gpu = model_gpu.forward(time_seqs_gpu[:, :-1], type_seqs_gpu[:, :-1], attention_mask_gpu[:, :-1, :-1])
        assert enc_output_gpu.shape == (
            sample_batch_data['time_seqs'].shape[0],
            sample_batch_data['time_seqs'].shape[1] - 1,
            sample_model_config.hidden_size
        )
        assert enc_output_gpu.device.type == "cuda"
