import math
from typing import Any, Optional, Tuple

import torch
from torch import nn

from new_ltpp.models.baselayer import EncoderLayer, MultiHeadAttention, ScaledSoftplus
from new_ltpp.models.neural_model import NeuralModel
from new_ltpp.shared_types import Batch
from new_ltpp.utils.attention import get_causal_attn_mask


class ANHP(NeuralModel):
    """Torch implementation of Attentive Neural Hawkes Process, ICLR 2022.
    https://arxiv.org/abs/2201.00044.
    Source code: https://github.com/yangalan123/anhp-andtt/blob/master/anhp/model/xfmr_nhp_fast.py
    """

    def __init__(
        self,
        *,
        use_norm: bool = True,
        time_emb_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize the model

        Args:
            model_config (new_ltpp.ModelConfig): config of model specs.
        """
        super(ANHP, self).__init__(**kwargs)

        self.d_model = self.hidden_size
        self.use_norm = use_norm
        self.d_time = time_emb_size

        self.div_term = torch.exp(
            torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)
        ).reshape(1, 1, -1)

        self.n_layers = num_layers
        self.n_head = num_heads

        # Créer les couches d'encodeur avec MultiHeadAttention
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    self.d_model + self.d_time,
                    MultiHeadAttention(
                        self.n_head,
                        self.d_model + self.d_time,
                        self.d_model + self.d_time,
                        self.dropout,
                    ),
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model + self.d_time)

        # La couche finale prend l'état du transformer + l'embedding du temps courant
        # Taille d'entrée = (d_model + d_time) + d_time car on concatène l'état (d_model + d_time) avec le temps cible (d_time)
        input_size_final = self.d_model + 2 * self.d_time
        self.inten_linear = nn.Linear(input_size_final, self.num_event_types)
        self.softplus = ScaledSoftplus(
            self.num_event_types
        )  # learnable mark-specific beta
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
        sample_time_emb: torch.Tensor,
        event_emb: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Architecture simplifiée et fonctionnelle.
        """
        # Dans cette version, event_emb contient déjà [Type + Time] de l'historique
        cur_layer_ = event_emb

        # 1. Encodage de l'historique
        for enc_layer in self.encoder_layers:
            cur_layer_ = enc_layer(cur_layer_, attention_mask)

        # 2. Concaténation avec le temps cible pour prédire l'intensité MAINTENANT
        # cur_layer_ : [Batch, Seq, Hidden] (Contexte historique)
        # sample_time_emb : [Batch, Seq, Hidden] (Temps où on veut l'intensité)

        final_state = torch.cat([cur_layer_, sample_time_emb], dim=-1)

        return final_state

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

    def forward(
        self,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call the model.

        Args:
            time_seqs: [batch_size, seq_len], sequences of timestamps.
            event_seqs: [batch_size, seq_len], sequences of event types.
            attention_mask: [seq_len, seq_len], masks for event sequences.
            sample_times: [batch_size, seq_len, num_samples]. Defaults to None.

        Returns:
            tensor: states at sampling times, [batch_size, seq_len, num_samples].
        """
        event_emb, time_emb, _ = self.seq_encoding(time_seqs, type_seqs)
        if sample_times is None:
            sample_time_emb = time_emb
        else:
            sample_time_emb = self.compute_temporal_embedding(sample_times)
        cur_layer_ = self.forward_pass(sample_time_emb, event_emb, attention_mask)

        return cur_layer_

    def loglike_loss(self, batch: Batch) -> Tuple[torch.Tensor, int]:
        """Compute the log-likelihood loss.

        Args:
            batch: batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """

        attn_mask = get_causal_attn_mask(batch.time_seqs.size(1), device=self.device)

        # 1. compute event-loglik
        # the prediction of last event has no label, so we proceed to the last but one
        # att mask => diag is False, not mask.
        enc_out = self.forward(
            batch.time_seqs[:, :-1],
            batch.type_seqs[:, :-1],
            attn_mask[:-1, :-1],
            batch.time_seqs[:, 1:],
        )

        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.layer_intensity(enc_out)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(batch.time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, seq_len = max_len - 1, num_sample, event_num]
        lambda_t_sample = self.compute_intensities_at_sample_dtimes(
            time_seqs=batch.time_seqs[:, :-1],
            time_delta_seqs=batch.time_delta_seqs[:, :-1],  # not used
            type_seqs=batch.type_seqs[:, :-1],
            sample_dtimes=sample_dtimes,
            attention_mask=attn_mask[:-1, :-1],
        )

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=batch.time_delta_seqs[:, 1:],
            seq_mask=batch.seq_non_pad_mask[:, 1:],
            type_seq=batch.type_seqs[:, 1:],
        )

        # compute loss to minimize
        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_states_at_sample_dtimes(
        self,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_dtimes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the states at sampling times.

        Args:
            time_seqs: [batch_size, seq_len], sequences of timestamps.
            type_seqs: [batch_size, seq_len], sequences of event types.
            attention_mask: [seq_len, seq_len], masks for event sequences.
            sample_dtimes: delta times in sampling.

        Returns:
            tensor: hiddens states at sampling times.
        """
        batch_size = type_seqs.size(0)
        seq_len = type_seqs.size(1)
        num_samples = sample_dtimes.size(-1)

        # [num_samples, batch_size, seq_len]
        sample_dtimes = sample_dtimes.permute([2, 0, 1])
        # [num_samples * batch_size, seq_len]
        _sample_dtimes = sample_dtimes.reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _types = type_seqs.expand(num_samples, -1, -1).reshape(
            num_samples * batch_size, -1
        )
        # [num_samples * batch_size, seq_len]
        _times = time_seqs.expand(num_samples, -1, -1).reshape(
            num_samples * batch_size, -1
        )

        # Correction 3 : Temps Absolu
        # On suppose que sample_dtimes a été reshape en [Batch*Samples, Seq]
        # _times aussi [Batch*Samples, Seq]

        # On calcule le temps absolu du sampling
        _sample_times_abs = _times + _sample_dtimes

        # On calcule l'embedding sur le temps ABSOLU
        encoder_output = self.forward(
            time_seqs=_times,
            type_seqs=_types,
            attention_mask=attention_mask,
            sample_times=_sample_times_abs,  # On passe le temps absolu ici
        )

        # [num_samples, batch_size, seq_len, hidden_size]
        hidden_size = encoder_output.size(-1)
        encoder_output = encoder_output.reshape(
            num_samples, batch_size, seq_len, hidden_size
        )

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = encoder_output.permute([1, 2, 0, 3])

        return encoder_output

    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the intensity at sampled delta times.

        Args:
            time_seqs (torch.Tensor): [batch_size, seq_len], sequences of timestamps.
            type_seqs (torch.Tensor): [batch_size, seq_len], sequences of event types.
            sample_dtimes (torch.Tensor): [batch_size, seq_len, num_sample], sampled delta time sequence.

        Returns:
            lambdas (torch.Tensor): [batch_size, seq_len, num_samples, event_num], intensities as sampled delta times.
        """

        attn_mask = get_causal_attn_mask(time_seqs.size(1), device=self.device)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_dtimes(
            time_seqs, type_seqs, attn_mask, sample_dtimes
        )

        if compute_last_step_only:
            # [batch_size, 1, num_samples, num_event_types]
            lambdas: torch.Tensor = self.layer_intensity(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas: torch.Tensor = self.layer_intensity(encoder_output)
        return lambdas
