'''
Unit tests for the THP model.
'''
import torch
import unittest
from easytpp.models.thp import THP
from easytpp.utils.model_utils import ModelConfig

class TestTHP(unittest.TestCase):
    def setUp(self):
        '''Set up for test cases.'''
        self.model_config = ModelConfig(
            model_name='thp',
            hidden_size=64,
            time_emb_size=16,
            use_ln=True,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            num_event_types=5,  # Including padding
            loss_type='loglike',
            num_mc_samples=20,
            device=torch.device("cpu")
        )
        self.model = THP(self.model_config).to(self.model_config.device)

        self.batch_size = 4
        self.seq_len = 10
        self.num_event_types = self.model_config.num_event_types -1 # Exclude padding for actual events

        # Dummy data
        self.time_seqs = torch.randn(self.batch_size, self.seq_len).abs().cumsum(dim=1).to(self.model_config.device)
        self.type_seqs = torch.randint(1, self.num_event_types + 1, (self.batch_size, self.seq_len)).to(self.model_config.device) # Event types > 0
        self.time_delta_seqs = torch.cat(
            (self.time_seqs[:, :1], self.time_seqs[:, 1:] - self.time_seqs[:, :-1]), dim=1
        ).to(self.model_config.device)
        self.batch_non_pad_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool).to(self.model_config.device)

        # Create attention mask (upper triangular for causality)
        attention_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=self.model_config.device), diagonal=1).unsqueeze(0)
        self.attention_mask = attention_mask.expand(self.batch_size, -1, -1).to(torch.bool)

        self.batch = (
            self.time_seqs,
            self.time_delta_seqs,
            self.type_seqs,
            self.batch_non_pad_mask,
            self.attention_mask
        )

    def test_thp_initialization(self):
        '''Test THP model initialization.'''
        self.assertIsNotNone(self.model, "Model should not be None after initialization")
        self.assertEqual(self.model.d_model, self.model_config.hidden_size)
        self.assertEqual(self.model.n_layers, self.model_config.num_layers)
        self.assertEqual(self.model.n_head, self.model_config.num_heads)
        self.assertEqual(self.model.num_event_types, self.model_config.num_event_types)

    def test_thp_forward_pass(self):
        '''Test the forward pass of the THP model.'''
        # Use seq_len - 1 for input to predict the next event
        enc_output = self.model.forward(self.time_seqs[:, :-1], self.type_seqs[:, :-1], self.attention_mask[:, :-1, :-1])
        self.assertEqual(enc_output.shape, (self.batch_size, self.seq_len - 1, self.model_config.hidden_size))

    def test_thp_loglike_loss(self):
        '''Test the log-likelihood loss computation.'''
        loss, num_events = self.model.loglike_loss(self.batch)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(num_events, torch.Tensor)
        self.assertEqual(loss.dim(), 0) # Scalar tensor
        self.assertGreaterEqual(num_events.item(), 0)

    def test_thp_compute_states_at_sample_times(self):
        '''Test computation of states at sampled times.'''
        enc_out = self.model.forward(self.time_seqs[:, :-1], self.type_seqs[:, :-1], self.attention_mask[:, :-1, :-1])
        sample_dtimes = self.model.make_dtime_loss_samples(self.time_delta_seqs[:, 1:])
        intensity_states = self.model.compute_states_at_sample_times(enc_out, sample_dtimes)
        expected_shape = (self.batch_size, self.seq_len - 1, self.model_config.num_mc_samples, self.model_config.num_event_types)
        self.assertEqual(intensity_states.shape, expected_shape)

    def test_thp_compute_intensities_at_sample_times(self):
        '''Test computation of intensities at sampled times.'''
        sample_dtimes = self.model.make_dtime_loss_samples(self.time_delta_seqs)
        # Note: THP's compute_intensities_at_sample_times expects full sequences for time_seqs, type_seqs, attention_mask
        lambdas = self.model.compute_intensities_at_sample_times(
            self.time_seqs,
            self.time_delta_seqs,
            self.type_seqs,
            sample_dtimes,
            attention_mask=self.attention_mask
        )
        expected_shape = (self.batch_size, self.seq_len, self.model_config.num_mc_samples, self.model_config.num_event_types)
        self.assertEqual(lambdas.shape, expected_shape)

    def test_thp_gradient_flow(self):
        '''Test if gradients flow correctly.'''
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optimizer.zero_grad()

        loss, _ = self.model.loglike_loss(self.batch)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                self.assertGreater(param.grad.abs().sum().item(), 0, f"Gradient for {name} is zero")

    def test_thp_batch_independence(self):
        '''Test if batch elements are processed independently.'''
        # Create a batch with two identical sequences and one different
        time_seqs_mod = self.time_seqs.clone()
        type_seqs_mod = self.type_seqs.clone()
        time_delta_seqs_mod = self.time_delta_seqs.clone()
        batch_non_pad_mask_mod = self.batch_non_pad_mask.clone()
        attention_mask_mod = self.attention_mask.clone()

        # Make the third sequence different
        if self.batch_size > 2:
            time_seqs_mod[2, :] = torch.rand_like(time_seqs_mod[2, :]).abs().cumsum(dim=0)
            type_seqs_mod[2, :] = torch.randint(1, self.num_event_types + 1, (self.seq_len,)).to(self.model_config.device)
            time_delta_seqs_mod[2, :] = torch.cat(
                (time_seqs_mod[2, :1], time_seqs_mod[2, 1:] - time_seqs_mod[2, :-1]), dim=0
            )

        batch_mod = (
            time_seqs_mod,
            time_delta_seqs_mod,
            type_seqs_mod,
            batch_non_pad_mask_mod,
            attention_mask_mod
        )

        self.model.eval() # Use eval mode to disable dropout for consistency
        loss_mod, _ = self.model.loglike_loss(batch_mod)

        # Process first two sequences (identical) individually
        batch1 = (
            time_seqs_mod[0:1],
            time_delta_seqs_mod[0:1],
            type_seqs_mod[0:1],
            batch_non_pad_mask_mod[0:1],
            attention_mask_mod[0:1]
        )
        loss1, _ = self.model.loglike_loss(batch1)

        batch2 = (
            time_seqs_mod[1:2],
            time_delta_seqs_mod[1:2],
            type_seqs_mod[1:2],
            batch_non_pad_mask_mod[1:2],
            attention_mask_mod[1:2]
        )
        loss2, _ = self.model.loglike_loss(batch2)

        # The sum of individual losses for identical inputs should be proportional if not equal
        # due to sum reduction in loss. We check if the per-event loss is similar.
        # For simplicity, we check if the first two elements of the batch output similar intermediate values.

        enc_out_mod = self.model.forward(time_seqs_mod[:, :-1], type_seqs_mod[:, :-1], attention_mask_mod[:, :-1, :-1])
        self.assertTrue(torch.allclose(enc_out_mod[0], enc_out_mod[1], atol=1e-6),
                        "Hidden states for identical sequences in a batch should be identical.")

        if self.batch_size > 2:
            self.assertFalse(torch.allclose(enc_out_mod[0], enc_out_mod[2], atol=1e-6),
                             "Hidden states for different sequences in a batch should be different.")

    def test_thp_state_dict_save_load(self):
        '''Test saving and loading the model state dictionary.'''
        model_new = THP(self.model_config).to(self.model_config.device)
        model_new.load_state_dict(self.model.state_dict())
        model_new.eval()
        self.model.eval()

        # Check if outputs are the same after loading state dict
        with torch.no_grad():
            enc_output_orig = self.model.forward(self.time_seqs[:, :-1], self.type_seqs[:, :-1], self.attention_mask[:, :-1, :-1])
            enc_output_new = model_new.forward(self.time_seqs[:, :-1], self.type_seqs[:, :-1], self.attention_mask[:, :-1, :-1])
        self.assertTrue(torch.allclose(enc_output_orig, enc_output_new, atol=1e-6))

    def test_thp_training_validation_modes(self):
        '''Test model behavior in training and validation modes (dropout).
           This test is more conceptual for THP as dropout is applied in EncoderLayer.
        '''
        # Check if dropout is active during training
        self.model.train()
        # To check dropout, we need to see if outputs vary for the same input
        # However, THP's forward pass is deterministic if weights are fixed.
        # A more robust test would involve checking if dropout layers are in train mode.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertTrue(module.training, "Dropout layer should be in training mode.")

        # Check if dropout is inactive during evaluation
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertFalse(module.training, "Dropout layer should be in eval mode.")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_thp_gpu_forward_pass(self):
        '''Test THP model forward pass on GPU.'''
        gpu_device = torch.device("cuda")
        model_gpu = THP(self.model_config).to(gpu_device)
        time_seqs_gpu = self.time_seqs.to(gpu_device)
        type_seqs_gpu = self.type_seqs.to(gpu_device)
        attention_mask_gpu = self.attention_mask.to(gpu_device)

        enc_output_gpu = model_gpu.forward(time_seqs_gpu[:, :-1], type_seqs_gpu[:, :-1], attention_mask_gpu[:, :-1, :-1])
        self.assertEqual(enc_output_gpu.shape, (self.batch_size, self.seq_len - 1, self.model_config.hidden_size))
        self.assertEqual(enc_output_gpu.device.type, "cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_thp_gpu_loglike_loss(self):
        '''Test THP model log-likelihood loss on GPU.'''
        gpu_device = torch.device("cuda")
        model_gpu = THP(self.model_config).to(gpu_device)

        time_seqs_gpu = self.time_seqs.to(gpu_device)
        time_delta_seqs_gpu = self.time_delta_seqs.to(gpu_device)
        type_seqs_gpu = self.type_seqs.to(gpu_device)
        batch_non_pad_mask_gpu = self.batch_non_pad_mask.to(gpu_device)
        attention_mask_gpu = self.attention_mask.to(gpu_device)

        batch_gpu = (
            time_seqs_gpu,
            time_delta_seqs_gpu,
            type_seqs_gpu,
            batch_non_pad_mask_gpu,
            attention_mask_gpu
        )

        loss_gpu, num_events_gpu = model_gpu.loglike_loss(batch_gpu)
        self.assertIsInstance(loss_gpu, torch.Tensor)
        self.assertEqual(loss_gpu.device.type, "cuda")
        self.assertEqual(num_events_gpu.device.type, "cuda")

if __name__ == '__main__':
    unittest.main()
