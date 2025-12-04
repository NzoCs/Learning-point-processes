import torch
import torch.nn as nn

from new_ltpp.configs.model_config import ModelConfig
from new_ltpp.shared_types import Batch
from new_ltpp.utils.attention import get_causal_attn_mask

from .baselayer import (
    EncoderLayer,
    MultiHeadAttention,
    ScaledSoftplus,
    TimeShiftedPositionalEncoding,
)
from .neural_model import NeuralModel


class SAHP(NeuralModel):
    """Torch implementation of Self-Attentive Hawkes Process, ICML 2020.
    Part of the code is collected from https://github.com/yangalan123/anhp-andtt/blob/master/sahp

    I slightly modify the original code because it is not stable.

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
        super(SAHP, self).__init__(**kwargs)
        self.d_model = self.hidden_size
        self.d_time = time_emb_size

        self.use_norm = use_norm

        # position vector, used for temporal encoding
        self.layer_position_emb = TimeShiftedPositionalEncoding(
            d_model=self.d_model, device=self.device
        )

        self.n_layers = num_layers
        self.n_head = num_heads

        # convert hidden vectors into a scalar
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(
            self.num_event_types
        )  # learnable mark-specific beta

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
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)

        # Equation (12): mu = GELU(h*W_mu)
        self.mu = nn.Sequential(
            nn.Linear(self.d_model, self.num_event_types, bias=False),
            nn.GELU(),
        )

        # Equation (13): eta = GELU(h*W_eta)
        self.eta = nn.Sequential(
            nn.Linear(self.d_model, self.num_event_types, bias=False),
            nn.GELU(),
        )

        # Equation (14): gamma = Softplus(h*W_gamma)
        self.gamma = nn.Sequential(
            nn.Linear(self.d_model, self.num_event_types, bias=False),
            nn.Softplus(),
        )

    def state_decay(self, encode_state, duration_t):
        """Equation (15), which computes the pre-intensity states

        Args:
            encode_state (tensor): [batch_size, seq_len, hidden_size].
            duration_t (tensor): [batch_size, seq_len, num_sample].

        Returns:
            tensor: hidden states at event times.
        """
        mu, eta, gamma = (
            self.mu(encode_state),
            self.eta(encode_state),
            self.gamma(encode_state),
        )

        # [batch_size, hidden_dim]
        states = mu + (eta - mu) * torch.exp(-gamma * duration_t)
        return states

    def forward(self, time_seqs, time_delta_seqs, event_seqs, key_padding_mask, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            event_seqs (tensor): [batch_size, seq_len], event type seqs.
            key_padding_mask (tensor): [batch_size, seq_len], key padding mask.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        type_embedding = self.layer_type_emb(event_seqs)
        position_embedding = self.layer_position_emb(time_seqs, time_delta_seqs)

        enc_output = type_embedding + position_embedding

        for enc_layer in self.stack_layers:
            enc_output = enc_layer(enc_output, key_padding_mask, attention_mask)
            if self.use_norm:
                enc_output = self.norm(enc_output)
        # [batch_size, seq_len, hidden_dim]
        return enc_output

    def loglike_loss(self, batch: Batch):
        """Compute the log-likelihood loss.

        Args:
            batch: batch input.

        Returns:
            list: loglike loss, num events.
        """
        time_seqs = batch.time_seqs
        time_delta_seqs = batch.time_delta_seqs
        type_seqs = batch.type_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask
        attention_mask = get_causal_attn_mask(time_seqs.size(1), device=self.device)

        enc_out = self.forward(
            time_seqs[:, :-1],
            time_delta_seqs[:, :-1],
            type_seqs[:, :-1],
            batch_non_pad_mask[:, :-1],
            attention_mask[:-1, :-1],
        )

        cell_t = self.state_decay(
            encode_state=enc_out, duration_t=time_delta_seqs[:, 1:, None]
        )

        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.softplus(cell_t)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(
            encode_state=enc_out, sample_dtimes=sample_dtimes
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

    def compute_states_at_sample_times(self, encode_state, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            encode_state (tensor): three tensors with each shape [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: [batch_size, seq_len, num_samples, hidden_size]， hidden state at each sampled time.
        """

        cell_states = self.state_decay(
            encode_state[:, :, None, :], sample_dtimes[:, :, :, None]
        )

        return cell_states

    def compute_intensities_at_sample_times(
        self,
        *,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.
            compute_last_step_only (bool): whether to compute only last step.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """
        attention_mask = get_causal_attn_mask(time_seqs.size(1), device=self.device)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(time_seqs, time_delta_seqs, type_seqs, None, attention_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        return lambdas
