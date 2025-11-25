from sympy import ask
import torch
import torch.nn as nn

from new_ltpp.shared_types import Batch
from new_ltpp.utils.attention import build_causal_attn_mask

from .baselayer import (
    EncoderLayer,
    MultiHeadAttention,
    ScaledSoftplus,
    TimePositionalEncoding,
)
from .neural_model import NeuralModel


class THP(NeuralModel):
    """Torch implementation of Transformer Hawkes Process, ICML 2020, https://arxiv.org/abs/2002.09291.
    Note: Part of the code is collected from https://github.com/yangalan123/anhp-andtt/tree/master/thp.
    """

    def __init__(
        self,
        *,
        use_norm: bool = True,
        time_emb_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        **kwargs,
    ):
        """Initialize the model

        Args:
            model_config (new_ltpp.ModelConfig): config of model specs.
        """
        super(THP, self).__init__(**kwargs)
        self.d_model = self.hidden_size
        self.d_time = time_emb_size
        self.use_norm = use_norm

        self.n_layers = num_layers
        self.n_head = num_heads

        self.layer_temporal_encoding = TimePositionalEncoding(
            self.d_model, device=self._device
        )

        self.factor_intensity_base = nn.Parameter(
            torch.empty([1, self.num_event_types], device=self._device)
        )
        self.factor_intensity_decay = nn.Parameter(
            torch.empty([1, self.num_event_types], device=self._device)
        )
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        # convert hidden vectors into event-type-sized vector
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(
            self.num_event_types
        )  # learnable mark-specific beta

        # Add MLP layer
        # Equation (5)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
        )

        self.stack_layers = nn.ModuleList(
            [
                EncoderLayer(
                    self.d_model,
                    MultiHeadAttention(
                        self.n_head,
                        self.d_model,
                        self.d_model,
                        self.dropout,
                        output_linear=False,
                    ),
                    use_residual=False,
                    feed_forward=self.feed_forward,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, time_seqs, type_seqs, key_padding_mask, attn_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        tem_enc = self.layer_temporal_encoding(time_seqs)
        enc_output = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        for enc_layer in self.stack_layers:
            enc_output += tem_enc
            enc_output = enc_layer(enc_output, key_padding_mask, attn_mask)

        return enc_output

    def loglike_loss(self, batch: Batch):
        """Compute the loglike loss.

        Args:
            batch: batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        time_seqs = batch.time_seqs
        time_delta_seqs = batch.time_delta_seqs
        type_seqs = batch.type_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask
        attn_mask = build_causal_attn_mask(len=time_seqs.size(1), device=self._device)

        enc_out = self.forward(
            time_seqs[:, :-1], 
            type_seqs[:, :-1], 
            key_padding_mask=~batch_non_pad_mask[:, :-1], 
            attn_mask=attn_mask[:-1, :-1]
        )

        # [batch_size, seq_len, num_event_types]
        # update time decay based on Equation (6)
        # [1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_event_types]
        intensity_states = (
            factor_intensity_decay * time_delta_seqs[:, 1:, None]
            + self.layer_intensity_hidden(enc_out)
            + factor_intensity_base
        )

        lambda_at_event = self.softplus(intensity_states)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(
            event_states=enc_out, sample_dtimes=sample_dtimes
        )
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=type_seqs[:, 1:],
        )

        # compute loss to minimize
        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = (
            factor_intensity_decay * sample_dtimes
            + self.layer_intensity_hidden(event_states)
            + factor_intensity_base
        )

        return intensity_states

    def compute_intensities_at_sample_times(
        self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs
    ):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        compute_last_step_only = kwargs.get("compute_last_step_only", False)
        attn_mask = build_causal_attn_mask(len=time_seqs.size(1), device=self._device)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(time_seqs, type_seqs, key_padding_mask=None, attn_mask=attn_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        return lambdas
