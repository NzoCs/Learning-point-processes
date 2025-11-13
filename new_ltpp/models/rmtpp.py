import math

import torch
from torch import nn
from torch.nn import functional as F

from new_ltpp.configs import ModelConfig
from new_ltpp.shared_types import Batch
from new_ltpp.models.basemodel import Model
from new_ltpp.models.neural_model import NeuralModel


class RMTPP(NeuralModel):
    """Torch implementation of Recurrent Marked Temporal Point Processes, KDD 2016.
    https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        num_event_types: int,
        dtime_max: float,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the model

        Args:
            model_config (new_ltpp.ModelConfig): config of model specs.
        """
        super(RMTPP, self).__init__(
            model_config,
            dtime_max=dtime_max,
            num_event_types=num_event_types,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        # self.hidden_size is now set in Model's __init__ via model_config.hidden_size

        self.layer_temporal_emb = nn.Linear(1, self.hidden_size)
        self.layer_rnn = nn.RNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=model_config.specs["num_layers"],
            nonlinearity="relu",
            batch_first=True,
        )

        self.hidden_to_intensity_logits = nn.Linear(
            self.hidden_size, self.num_event_types
        )
        self.b_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.w_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        nn.init.xavier_normal_(self.b_t)
        nn.init.xavier_normal_(self.w_t)

    def evolve_and_get_intentsity(self, right_hiddens_BNH, dts_BNG):
        """
        Eq.11 that computes intensity.
        """

        past_influence_BNGM = self.hidden_to_intensity_logits(
            right_hiddens_BNH[..., None, :]
        )
        intensity_BNGM = (
            (
                past_influence_BNGM
                + self.w_t[None, None, :] * dts_BNG[..., None]
                + self.b_t[None, None, :]
            )
            .clamp(max=math.log(1e5))
            .exp()
        )

        return intensity_BNGM

    def forward(self, batch: Batch):
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
        ts_BN = batch.time_seqs
        dt_BN = batch.time_delta_seqs
        marks_BN = batch.type_seqs

        mark_emb_BNH = self.layer_type_emb(marks_BN)
        time_emb_BNH = self.layer_temporal_emb(ts_BN[..., None])
        rnn_input = mark_emb_BNH + time_emb_BNH
        right_hiddens_BNH, _ = self.layer_rnn(rnn_input)
        left_intensity_B_Nm1_G_M = self.evolve_and_get_intentsity(
            right_hiddens_BNH[:, :-1, :], dt_BN[:, 1:][..., None]
        )
        left_intensity_B_Nm1_M = left_intensity_B_Nm1_G_M.squeeze(-2)
        return left_intensity_B_Nm1_M, right_hiddens_BNH

    def loglike_loss(self, batch: Batch):
        """Compute the log-likelihood loss.

        Args:
            batch: batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        ts_BN = batch.time_seqs
        dts_BN = batch.time_delta_seqs
        marks_BN = batch.type_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask

        # Call forward with batch directly
        left_intensity_B_Nm1_M, right_hiddens_BNH = self.forward(batch)
        right_hiddens_B_Nm1_H = right_hiddens_BNH[
            ..., :-1, :
        ]  # discard right limit at t_N for logL

        dts_sample_B_Nm1_G = self.make_dtime_loss_samples(dts_BN[:, 1:])
        intensity_dts_B_Nm1_G_M = self.evolve_and_get_intentsity(
            right_hiddens_B_Nm1_H, dts_sample_B_Nm1_G
        )

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=left_intensity_B_Nm1_M,
            lambdas_loss_samples=intensity_dts_B_Nm1_G_M,
            time_delta_seq=dts_BN[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=marks_BN[:, 1:],
        )

        # compute loss to minimize
        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_intensities_at_sample_times(
        self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs
    ):
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

        compute_last_step_only = kwargs.get("compute_last_step_only", False)

        batch = Batch(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            seq_non_pad_mask=torch.empty(0, dtype=torch.bool),
            attention_mask=torch.empty(0, device=self.device),
        )

        _, right_hiddens_BNH = self.forward(batch)

        if compute_last_step_only:
            sampled_intensities = self.evolve_and_get_intentsity(
                right_hiddens_BNH[:, -1:, :], sample_dtimes[:, -1:, :]
            )
        else:
            sampled_intensities = self.evolve_and_get_intentsity(
                right_hiddens_BNH, sample_dtimes
            )  # shape: [B, N, G, M]
        return sampled_intensities
