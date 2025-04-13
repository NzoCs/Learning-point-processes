import torch
from torch import nn
import math
from torch.nn import functional as F

from easy_tpp.models.basemodel import BaseModel

class RMTPP(BaseModel):
    """Torch implementation of Recurrent Marked Temporal Point Processes, KDD 2016.
    https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(RMTPP, self).__init__(model_config)

        self.layer_temporal_emb = nn.Linear(1, self.hidden_size)
        self.layer_rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                num_layers=1, nonlinearity='relu', batch_first=True) 

        self.hidden_to_intensity_logits = nn.Linear(self.hidden_size, self.num_event_types)
        self.b_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.w_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        nn.init.xavier_normal_(self.b_t)
        nn.init.xavier_normal_(self.w_t)


    def evolve_and_get_intentsity(self, right_hiddens_BNH, dts_BNG):
        """
        Eq.11 that computes intensity.
        """
    
        past_influence_BNGM = self.hidden_to_intensity_logits(right_hiddens_BNH[..., None, :])
        intensity_BNGM = (past_influence_BNGM + self.w_t[None, None, :] * dts_BNG[..., None]
                        + self.b_t[None, None, :]).clamp(max=math.log(1e5)).exp()
        
        return intensity_BNGM

    def forward(self, batch):
        """
        Suppose we have inputs with original sequence length N+1
        ts: [t0, t1, ..., t_N]
        dts: [0, t1 - t0, t2 - t1, ..., t_N - t_{N-1}]
        marks: [k0, k1, ..., k_N] (k0 and kN could be padded marks if t0 and tN correspond to left and right windows)

        Return:
            left limits of *intensity* at [t_1, ..., t_N] of shape: (batch_size, seq_len - 1, hidden_dim)
            right limits of *hidden states* [t_0, ..., t_{N-1}, t_N] of shape: (batch_size, seq_len, hidden_dim)
            We need the right limit of t_N to sample continuation.
        """

        t_BN, dt_BN, marks_BN, _, _ = batch
        mark_emb_BNH = self.layer_type_emb(marks_BN)
        time_emb_BNH = self.layer_temporal_emb(t_BN[..., None])
        right_hiddens_BNH, _ = self.layer_rnn(mark_emb_BNH + time_emb_BNH)
        left_intensity_B_Nm1_G_M = self.evolve_and_get_intentsity(right_hiddens_BNH[:, :-1, :], dt_BN[:, 1:][...,None])
        left_intensity_B_Nm1_M = left_intensity_B_Nm1_G_M.squeeze(-2)
        return left_intensity_B_Nm1_M, right_hiddens_BNH
    
    def compute_loglikelihood(self, lambda_at_event, lambda_at_0, seq_mask, type_seq):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambda_at_0 (tensor) : [batch_size,seq_len,num_event_types], unmasked intensity at time 0
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids, with padded events having a mark of self.pad_token_id

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """

        # First, add an epsilon to every marked intensity for stability
        lambda_at_event = lambda_at_event + self.eps
        #lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        #total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )
        
        w_t_11M = self.eps + self.w_t
        non_event_ll_unmasked=(1/w_t_11M)*(lambda_at_event-lambda_at_0)
        non_event_ll=non_event_ll_unmasked.sum(-1)*seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events

    def loglike_loss(self, batch):
        """Compute the log-likelihood loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        ts_BN, dts_BN, marks_BN, batch_non_pad_mask, _ = batch

        # compute left intensity and hidden states at event time
        # left limits of intensity at [t_1, ..., t_N]
        # right limits of hidden states at [t_0, ..., t_{N-1}, t_N]
        left_intensity_B_Nm1_M, right_hiddens_BNH = self.forward((ts_BN, dts_BN, marks_BN, None, None))

        past_influence_B_Nm1_M = self.hidden_to_intensity_logits(right_hiddens_BNH[:,:-1,:])
        
        intensity_at_0_B_Nm1_M = (past_influence_B_Nm1_M
                         + self.b_t[None, :]).clamp(max=math.log(1e5)).exp()

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=left_intensity_B_Nm1_M,
            lambda_at_0=intensity_at_0_B_Nm1_M,
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=marks_BN[:, 1:]
        )

        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events



    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seq (tensor): [batch_size, seq_len], times seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        _input = time_seqs, time_delta_seqs, type_seqs, None, None
        _, right_hiddens_BNH = self.forward(_input)

        if compute_last_step_only:
            sampled_intensities = self.evolve_and_get_intentsity(right_hiddens_BNH[:, -1:, :], sample_dtimes[:, -1:, :])
        else:
            sampled_intensities = self.evolve_and_get_intentsity(right_hiddens_BNH, sample_dtimes)  # shape: [B, N, G, M]
        return sampled_intensities