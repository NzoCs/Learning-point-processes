import math
from typing import Optional, Tuple

import torch
from torch import nn

from new_ltpp.configs import ModelConfig
from new_ltpp.models.baselayer import EncoderLayer, MultiHeadAttention, ScaledSoftplus
from new_ltpp.models.neural_model import NeuralModel
from new_ltpp.shared_types import Batch
from new_ltpp.utils.attention import build_attention_mask_from_seq_mask


class AttNHP(NeuralModel):
    """Torch implementation of Attentive Neural Hawkes Process, ICLR 2022.
    https://arxiv.org/abs/2201.00044.
    Source code: https://github.com/yangalan123/anhp-andtt/blob/master/anhp/model/xfmr_nhp_fast.py
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        num_event_types: int,
        dtime_max: float,
        hidden_size: int,
        dropout: float,
        use_norm: bool = True,
        time_emb_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
    ):
        """Initialize the model

        Args:
            model_config (new_ltpp.ModelConfig): config of model specs.
        """
        super(AttNHP, self).__init__(
            model_config,
            dtime_max=dtime_max,
            num_event_types=num_event_types,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.d_model = self.hidden_size
        self.use_norm = use_norm
        self.d_time = time_emb_size

        self.div_term = torch.exp(
            torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)
        ).reshape(1, 1, -1)

        self.n_layers = num_layers
        self.n_head = num_heads

        # Type annotation for better type checking
        self.heads: nn.ModuleList = nn.ModuleList()
        for i in range(self.n_head):
            head_layers = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_model + self.d_time,
                        MultiHeadAttention(
                            1,
                            self.d_model + self.d_time,
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
            self.heads.append(head_layers)

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)
        self.inten_linear = nn.Linear(self.d_model * self.n_head, self.num_event_types)
        self.softplus = ScaledSoftplus(
            self.num_event_types
        )  # learnable mark-specific beta
        self.layer_event_emb = nn.Linear(self.d_model + self.d_time, self.d_model)
        self.layer_intensity = nn.Sequential(self.inten_linear, self.softplus)
        self.eps = torch.finfo(torch.float32).eps

    def compute_temporal_embedding(self, time: torch.Tensor) -> torch.Tensor:
        """Compute the temporal embedding.

        Args:
            time: [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, emb_size].
        """
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)

        return pe

    def forward_pass(
        self,
        init_cur_layer: torch.Tensor,
        time_emb: torch.Tensor,
        sample_time_emb: torch.Tensor,
        event_emb: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> torch.Tensor:
        """update the structure sequentially.

        Args:
            init_cur_layer: [batch_size, seq_len, hidden_size]
            time_emb: [batch_size, seq_len, hidden_size]
            sample_time_emb: [batch_size, seq_len, hidden_size]
            event_emb: [batch_size, seq_len, hidden_size]
            combined_mask: [batch_size, seq_len, hidden_size]

        Returns:
            tensor: [batch_size, seq_len, hidden_size*2]
        """
        cur_layers = []
        seq_len = event_emb.size(1)
        for head_i in range(self.n_head):
            # [batch_size, seq_len, hidden_size]
            cur_layer_ = init_cur_layer
            # Get the head's layer list - cast to ModuleList for type checker
            head_layers = self.heads[head_i]
            assert isinstance(head_layers, nn.ModuleList)
            
            for layer_i in range(self.n_layers):
                # each layer concats the temporal emb
                # [batch_size, seq_len, hidden_size*2]
                layer_ = torch.cat([cur_layer_, sample_time_emb], dim=-1)
                # make combined input from event emb + layer emb
                # [batch_size, seq_len*2, hidden_size*2]
                _combined_input = torch.cat([event_emb, layer_], dim=1)
                enc_layer = head_layers[layer_i]
                # compute the output
                enc_output = enc_layer(_combined_input, combined_mask)

                # the layer output
                # [batch_size, seq_len, hidden_size]
                _cur_layer_ = enc_output[:, seq_len:, :]
                # add residual connection
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_

                # event emb
                event_emb = torch.cat([enc_output[:, :seq_len, :], time_emb], dim=-1)

                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    def seq_encoding(
        self, time_seqs: torch.Tensor, event_seqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the sequence.

        Args:
            time_seqs: time seqs input, [batch_size, seq_len].
            event_seqs: event type seqs input, [batch_size, seq_len].

        Returns:
            tuple: event embedding, time embedding and type embedding.
        """
        # [batch_size, seq_len, hidden_size]
        time_emb = self.compute_temporal_embedding(time_seqs)
        # [batch_size, seq_len, hidden_size]
        type_emb = torch.tanh(self.layer_type_emb(event_seqs.long()))
        # [batch_size, seq_len, hidden_size*2]
        event_emb = torch.cat([type_emb, time_emb], dim=-1)

        return event_emb, time_emb, type_emb

    def make_layer_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create a tensor to do masking on layers.

        Args:
            attention_mask: mask for attention operation, [batch_size, seq_len, seq_len]

        Returns:
            tensor: aim to keep the current layer, the same size of attention mask
            a diagonal matrix, [batch_size, seq_len, seq_len]
        """
        # [batch_size, seq_len, seq_len]
        layer_mask = (
            (torch.eye(attention_mask.size(1), device=self.device) < 1)
            .unsqueeze(0)
            .expand_as(attention_mask)
        )
        return layer_mask

    def make_combined_att_mask(
        self, attention_mask: torch.Tensor, layer_mask: torch.Tensor
    ) -> torch.Tensor:
        """Combined attention mask and layer mask.

        Args:
            attention_mask: mask for attention operation, [batch_size, seq_len, seq_len]
            layer_mask: mask for other layers, [batch_size, seq_len, seq_len]

        Returns:
            tensor: [batch_size, seq_len * 2, seq_len * 2]
        """
        # [batch_size, seq_len, seq_len * 2]
        combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # [batch_size, seq_len, seq_len * 2]
        contextual_mask = torch.cat(
            [attention_mask, torch.ones_like(layer_mask)], dim=-1
        )
        # [batch_size, seq_len * 2, seq_len * 2]
        combined_mask = torch.cat([contextual_mask, combined_mask], dim=1)
        return combined_mask

    def forward(
        self,
        time_seqs: torch.Tensor,
        event_seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call the model.

        Args:
            time_seqs: [batch_size, seq_len], sequences of timestamps.
            event_seqs: [batch_size, seq_len], sequences of event types.
            attention_mask: [batch_size, seq_len, seq_len], masks for event sequences.
            sample_times: [batch_size, seq_len, num_samples]. Defaults to None.

        Returns:
            tensor: states at sampling times, [batch_size, seq_len, num_samples].
        """
        event_emb, time_emb, type_emb = self.seq_encoding(time_seqs, event_seqs)
        init_cur_layer = torch.zeros_like(type_emb)
        layer_mask = self.make_layer_mask(attention_mask)
        if sample_times is None:
            sample_time_emb = time_emb
        else:
            sample_time_emb = self.compute_temporal_embedding(sample_times)
        combined_mask = self.make_combined_att_mask(attention_mask, layer_mask)
        cur_layer_ = self.forward_pass(
            init_cur_layer, time_emb, sample_time_emb, event_emb, combined_mask
        )

        return cur_layer_

    def loglike_loss(self, batch: Batch) -> Tuple[torch.Tensor, int]:
        """Compute the log-likelihood loss.

        Args:
            batch: batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        time_seqs = batch.time_seqs
        time_delta_seqs = batch.time_delta_seqs
        type_seqs = batch.type_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask
        attention_mask = build_attention_mask_from_seq_mask(batch.seq_non_pad_mask)
        # 1. compute event-loglik
        # the prediction of last event has no label, so we proceed to the last but one
        # att mask => diag is False, not mask.
        enc_out = self.forward(
            time_seqs[:, :-1],
            type_seqs[:, :-1],
            attention_mask[:, :-1, :-1],
            time_seqs[:, 1:],
        )
        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.layer_intensity(enc_out)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        temp_time = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # [batch_size, seq_len, num_sample]
        sample_times = temp_time + time_seqs[:, :-1].unsqueeze(-1)

        # 2.2 compute intensities at sampled times
        # [batch_size, seq_len = max_len - 1, num_sample, event_num]
        lambda_t_sample = self.compute_intensities_at_sample_times(
            time_seqs[:, :-1],
            time_delta_seqs[:, :-1],  # not used
            type_seqs[:, :-1],
            sample_times,
            attention_mask=attention_mask[:, :-1, :-1],
        )

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

    def compute_states_at_sample_times(
        self,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_times: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the states at sampling times.

        Args:
            time_seqs: [batch_size, seq_len], sequences of timestamps.
            type_seqs: [batch_size, seq_len], sequences of event types.
            attention_mask: [batch_size, seq_len, seq_len], masks for event sequences.
            sample_times: delta times in sampling.

        Returns:
            tensor: hiddens states at sampling times.
        """
        batch_size = type_seqs.size(0)
        seq_len = type_seqs.size(1)
        num_samples = sample_times.size(-1)

        # [num_samples, batch_size, seq_len]
        sample_times = sample_times.permute((2, 0, 1))
        # [num_samples * batch_size, seq_len]
        _sample_time = sample_times.reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _types = type_seqs.expand(num_samples, -1, -1).reshape(
            num_samples * batch_size, -1
        )
        # [num_samples * batch_size, seq_len]
        _times = time_seqs.expand(num_samples, -1, -1).reshape(
            num_samples * batch_size, -1
        )
        # [num_samples * batch_size, seq_len]
        _attn_mask = (
            attention_mask.unsqueeze(0)
            .expand(num_samples, -1, -1, -1)
            .reshape(num_samples * batch_size, seq_len, seq_len)
        )
        # [num_samples * batch_size, seq_len, hidden_size]
        encoder_output = self.forward(_times, _types, _attn_mask, _sample_time)

        # [num_samples, batch_size, seq_len, hidden_size]
        encoder_output = encoder_output.reshape(num_samples, batch_size, seq_len, -1)
        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = encoder_output.permute((1, 2, 0, 3))
        return encoder_output

    def compute_intensities_at_sample_times(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the intensity at sampled times.

        Args:
            time_seqs: [batch_size, seq_len], sequences of timestamps.
            time_delta_seqs: [batch_size, seq_len], sequences of delta times.
            type_seqs: [batch_size, seq_len], sequences of event types.
            sample_dtimes: [batch_size, seq_len, num_sample], sampled time delta sequence.

        Returns:
            tensor: intensities as sampled_dtimes, [batch_size, seq_len, num_samples, event_num].
        """
        attention_mask = kwargs.get("attention_mask", None)
        compute_last_step_only = kwargs.get("compute_last_step_only", False)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = (
                torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
                .unsqueeze(0)
                .to(type_seqs.device)
            )
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        if sample_dtimes.size()[1] < time_seqs.size()[1]:
            # we pass sample_dtimes for last time step here
            # we do a temp solution
            # [batch_size, seq_len, num_samples]
            sample_dtimes = time_seqs[:, :, None] + torch.tile(
                sample_dtimes, [1, time_seqs.size()[1], 1]
            )

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(
            time_seqs, type_seqs, attention_mask, sample_dtimes
        )

        if compute_last_step_only:
            lambdas = self.layer_intensity(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.layer_intensity(encoder_output)
        return lambdas
