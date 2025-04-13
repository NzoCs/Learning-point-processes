import torch
from torchtyping import TensorType

from easy_tpp.synthetic_data_generator.base_generator import BaseGenerator

class SelfCorrecting(BaseGenerator) :
     
     def __init__(self, gen_config, save_dir):
          super(SelfCorrecting, self).__init__(gen_config, save_dir)
          
          model_config = gen_config.model_config
          self.mu = model_config.mu
          self.alpha = model_config.alpha
          self.num_mark = self.mu.size(0)
     
     def compute_intensities_at_sample_times(
        self,
        batch : tuple[TensorType["batch_size", "seq_len"], TensorType["batch_size", "seq_len"], TensorType["batch_size", "seq_len"], TensorType["batch_size", "seq_len"]],
        sample_batch : tuple[TensorType["batch_size", "sample_len", "num_sample"], TensorType["batch_size", "sample_len", "num_sample"]],
        **kwargs
        ) -> TensorType["batch_size", "sample_len", "num_sample", "num_mark"] : 
          
          mu = self.mu
          alpha = self.alpha
          num_mark = self.num_mark
          
          batch_size, sample_len, num_sample = sample_batch[0].size()
          _, seq_len = batch[0].size()
          
          time_seq, time_delta_seq, event_seq, batch_non_pad_mask,_ = batch
          sampled_time_seq, sampled_time_delta_seq = sample_batch
          
          type_seq_expanded = event_seq[:, None, :].expand(batch_size, sample_len, num_sample, num_mark, seq_len)
          time_seq_expanded = time_seq[:, None, None, None, :].repeat(1, sample_len, num_sample, num_mark, 1)
        
          mask1 = torch.where(time_seq_expanded <= sampled_time_seq, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
          mark_tensor = torch.arange(num_mark, device=self.device).float().view(1, 1, 1, num_mark, 1)
          mask2 = torch.where(type_seq_expanded == mark_tensor, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
          mask = mask1 * mask2 * batch_non_pad_mask[:, None, None, None, :]
          
          time_contributions = mask.sum(dim=-1).view(batch_size, sample_len, num_sample, 1, num_mark) * alpha.view(1, 1, 1, num_mark, num_mark) 
          intensities = torch.exp(mu * sampled_time_seq.view(batch_size, sample_len, num_sample, 1) + time_contributions.sum(dim=-1))
          
          return intensities