"""Attentive Neural Hawkes Network (ANHN) implementation."""

from typing import TypedDict, Tuple

import torch
from torch import nn

# Assurez-vous que ces imports correspondent à votre structure de projet
from new_ltpp.models.baselayer import MultiHeadAttention
from new_ltpp.models.neural_model import NeuralModel
from new_ltpp.shared_types import Batch
from new_ltpp.utils.attention import get_causal_attn_mask


class ANHNOutput(TypedDict):
    """Output of the forward pass of the ANHN model."""

    imply_lambdas: torch.Tensor  # Intensités aux points d'échantillonnage
    intensity_base: torch.Tensor  # Mu
    intensity_alpha: torch.Tensor  # Alpha
    intensity_delta: torch.Tensor  # Delta
    sample_elapsed_times: torch.Tensor  # Temps écoulés avec échantillons
    base_elapsed_times: (
        torch.Tensor
    )  # Temps écoulés entre événements (sans échantillons)


class ANHN(NeuralModel):
    """Torch implementation of Attentive Neural Hawkes Network, IJCNN 2021.
    http://arxiv.org/abs/2211.11758
    """

    def __init__(
        self,
        *,
        num_layers: int = 2,
        num_heads: int = 2,
        use_norm: bool = True,
        time_emb_size: int = 32,
        **kwargs,
    ):
        super(ANHN, self).__init__(**kwargs)

        self.d_time = time_emb_size
        self.use_norm = use_norm
        self.n_layers = num_layers
        self.n_head = num_heads

        # Couche RNN pour capturer la dépendance séquentielle locale
        self.layer_rnn = nn.LSTM(
            input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True
        )

        # Paramètres potentiellement inutilisés (hérités du papier original)
        self.lambda_w = torch.empty([self.hidden_size, self.num_event_types])
        self.lambda_b = torch.empty([self.num_event_types, 1])
        nn.init.xavier_normal_(self.lambda_w)
        nn.init.xavier_normal_(self.lambda_b)

        # Couche pour calculer le taux de décroissance (Delta)
        self.layer_time_delta = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size), nn.Softplus()
        )

        # Couche pour l'intensité de base (Mu)
        self.layer_base_intensity = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.Sigmoid()
        )

        # Attention Multi-Têtes pour les dépendances à long terme
        self.layer_att = MultiHeadAttention(
            self.n_head,
            self.hidden_size,
            self.hidden_size,
            self.dropout,
        )

        # Projection finale vers l'espace des types d'événements
        self.layer_intensity = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_event_types), nn.Softplus()
        )

        self.softplus = nn.Softplus()

    def forward(
        self,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ANHNOutput:
        """Call the model."""

        # [batch_size, seq_len, hidden_size]
        # Note: self.layer_type_emb doit être défini dans NeuralModel
        event_emb: torch.Tensor = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        rnn_output: torch.Tensor;
        rnn_output, _ = self.layer_rnn(event_emb)


        # --- 1. Calcul des paramètres du Hawkes (Mu, Alpha, Delta) ---

        # [batch_size, seq_len, hidden_size]
        # mu in Equation (3)
        intensity_base = self.layer_base_intensity(rnn_output)

        # [batch_size, seq_len, seq_len]
        _, att_weight = self.layer_att(
            rnn_output,
            rnn_output,
            rnn_output,
            attn_mask=attention_mask,
            output_weight=True,
        )

        # [batch_size, seq_len, seq_len, hidden_size]
        # Alpha: pondération par l'attention
        intensity_alpha = att_weight[:, :, :, None] * rnn_output[:, None, :, :]

        # Compute Delta
        max_len = event_emb.size(1)

        # Construction des paires pour le calcul de delta
        # [batch_size, seq_len, seq_len, hidden_size]
        left = rnn_output[:, None, :, :].expand(-1, max_len, -1, -1)
        right = rnn_output[:, :, None, :].expand(-1, -1, max_len, -1)

        # [batch_size, seq_len, seq_len, hidden_size * 2]
        cur_prev_concat = torch.cat([left, right], dim=-1)

        # [batch_size, seq_len, seq_len, hidden_size]
        intensity_delta = self.layer_time_delta(cur_prev_concat)

        # --- 2. Calcul des temps écoulés ---

        # [batch_size, seq_len, seq_len, num_samples]
        sample_elapsed_times, base_elapsed_times = self.compute_elapsed_times(
            time_seqs, sample_dtimes
        )

        # --- 3. Calcul de l'état caché aux temps échantillonnés ---

        # [batch_size, seq_len, num_samples, hidden_size]
        imply_lambdas = self.compute_states_at_sample_dtimes(
            intensity_base, intensity_alpha, intensity_delta, sample_elapsed_times
        )

        return ANHNOutput(
            imply_lambdas=imply_lambdas,
            intensity_base=intensity_base,
            intensity_alpha=intensity_alpha,
            intensity_delta=intensity_delta,
            sample_elapsed_times=sample_elapsed_times,
            base_elapsed_times=base_elapsed_times,
        )

    def compute_elapsed_times(
        self, time_seqs: torch.Tensor, sample_dtimes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute elapsed times efficiently.
        Returns:
            sample_elapsed_times: [batch, seq_len, seq_len, num_samples]
            base_elapsed_times: [batch, seq_len, seq_len]
        """
        # [batch_size, seq_len, 1] (Cibles j)
        target_times = time_seqs.unsqueeze(dim=-1)

        # [batch_size, 1, seq_len] (Sources i)
        source_times = time_seqs.unsqueeze(dim=-2)

        # [batch_size, seq_len, seq_len]
        # Broadcasting: (j, 1) - (1, i) -> (j, i)
        base_elapsed_times = target_times - source_times

        # Alignement des dimensions pour l'ajout du temps échantillonné
        # base:   [B, L, L]    -> [B, L, L, 1]
        # sample: [B, L, S]    -> [B, L, 1, S] (On insère la dim source en pos 2)

        # [B, L, L, S]
        elapsed_times = base_elapsed_times.unsqueeze(-1) + sample_dtimes.unsqueeze(2)

        # IMPORTANT: Remplacer les temps négatifs (futur) par l'infini
        # exp(-delta * inf) = 0. Si on met 0, exp(0)=1 (influence max), ce qui est faux.
        # On utilise 1e9 comme proxy pour l'infini pour la stabilité numérique.
        sample_elapsed_times = elapsed_times.masked_fill(elapsed_times < 0, 1e9)

        # De même pour base_elapsed_times (pour le calcul aux événements)
        # Mais ici on garde la forme [B, L, L, 1] pour la compatibilité future
        base_elapsed_masked = base_elapsed_times.unsqueeze(-1).masked_fill(
            base_elapsed_times.unsqueeze(-1) < 0, 1e9
        )

        return sample_elapsed_times, base_elapsed_masked

    def compute_states_at_sample_dtimes(
        self,
        intensity_base: torch.Tensor,
        intensity_alpha: torch.Tensor,
        intensity_delta: torch.Tensor,
        sample_elapsed_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the hidden states at sampled times (Fully Vectorized).
        Args:
            intensity_base (mu): [batch_size, seq_len, hidden_size]
            intensity_alpha (alpha): [batch_size, seq_len, seq_len, hidden_size]
            intensity_delta (delta): [batch_size, seq_len, seq_len, hidden_size]
            sample_elapsed_times (t - t_i): [batch_size, seq_len, seq_len, num_samples]
        Returns:
            imply_lambdas (tensor): [batch_size, seq_len, num_sample, hidden_size]
            Latent intensity at sampled times.
        """
        # 1. Alignement des dimensions pour le Broadcasting
        # On vise la forme : [B, Target(L), Source(L), Sample(S), Hidden(H)]

        # [B, L, L, H] -> [B, L, L, 1, H]
        alpha = intensity_alpha.unsqueeze(3)
        delta = intensity_delta.unsqueeze(3)

        # [B, L, L, S] -> [B, L, L, S, 1]
        elapsed = sample_elapsed_times.unsqueeze(4)

        # 2. Calcul du noyau (Kernel) : alpha * exp(-delta * t)
        # exp( - [B, L, L, 1, H] * [B, L, L, S, 1] ) -> [B, L, L, S, H]
        decay_term = torch.exp(-delta * elapsed)

        # [B, L, L, 1, H] * [B, L, L, S, H] -> [B, L, L, S, H]
        weighted_decay = alpha * decay_term

        # 3. Somme sur l'historique (l'axe Source, dim=2)
        # [B, L, S, H]
        history_effect = torch.sum(weighted_decay, dim=2)

        # 4. Ajout de l'intensité de base (Mu)
        # [B, L, H] -> [B, L, 1, H] pour matcher [B, L, S, H]
        imply_lambdas = intensity_base.unsqueeze(2) + history_effect

        return imply_lambdas

    def loglike_loss(self, batch: Batch):
        """Compute the loglikelihood loss."""

        # Prep inputs
        time_seqs = batch.time_seqs[:, 1:]
        time_delta_seqs = batch.time_delta_seqs[:, 1:]
        type_seqs = batch.type_seqs[:, :-1]

        # 1. Samples for integral
        # [batch_size, seq_len, num_mc_samples]
        sample_dtimes_random = self.make_dtime_loss_samples(time_delta_seqs)

        # 2. Add zero for event time intensity (to compute lambda at t_i)
        # [batch_size, seq_len, 1]
        zeros = torch.zeros_like(time_delta_seqs).unsqueeze(-1)
        # [batch_size, seq_len, num_mc_samples + 1]
        all_sample_dtimes = torch.cat([sample_dtimes_random, zeros], dim=-1)

        # 3. Compute all intensities in one go
        # [batch_size, seq_len, num_mc_samples + 1, num_event_types]
        all_lambdas = self.compute_intensities_at_sample_dtimes(
            time_delta_seqs=time_delta_seqs,
            time_seqs=time_seqs,
            type_seqs=type_seqs,
            sample_dtimes=all_sample_dtimes,
        )

        # 4. Split results
        # [batch_size, seq_len, num_mc_samples, num_event_types]
        lambda_t_sample = all_lambdas[:, :, :-1, :]
        # [batch_size, seq_len, num_event_types]
        lambda_at_event = all_lambdas[:, :, -1, :]

        # 5. Compute LL
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs,
            seq_mask=batch.seq_non_pad_mask[:, 1:],
            type_seq=batch.type_seqs[:, 1:],
        )

        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_delta_seqs: torch.Tensor,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the intensity at sampled times.

        Args:
            sample_dtimes (tensor): [batch_size, seq_len, num_sample],
            delta times at which to sample intensities.

        Returns:
            lambdas (tensor): [batch_size, seq_len, num_samples, event_num]
            Intensities as sampled_dtimes.
        """

        attn_mask = get_causal_attn_mask(time_delta_seqs.size(1), device=self.device)

        # [batch_size, seq_len, num_samples]
        outputs = self.forward(
            time_seqs=time_seqs,
            type_seqs=type_seqs,
            sample_dtimes=sample_dtimes,
            attention_mask=attn_mask,
        )

        # [batch_size, seq_len, num_samples, hidden_size]
        imply_lambdas = outputs["imply_lambdas"]

        if compute_last_step_only:
            imply_lambdas = imply_lambdas[:, -1:, :, :]

        # [batch_size, seq_len, num_samples, num_event_types]
        lambdas = self.layer_intensity(imply_lambdas)
        return lambdas
