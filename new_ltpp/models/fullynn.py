from typing import Tuple

import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F

from new_ltpp.shared_types import Batch

from .neural_model import NeuralModel


class CumulHazardFunctionNetwork(nn.Module):
    """Cumulative Hazard Function Network
    ref: https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_event_types: int,
        num_mlp_layers: int = 3,
        proper_marked_intensities: bool = True,
    ):

        super(CumulHazardFunctionNetwork, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.num_event_types = num_event_types
        self.proper_marked_intensities = proper_marked_intensities

        # transform inter-event time embedding
        self.layer_dense_1 = nn.Linear(in_features=1, out_features=hidden_size)

        # concat rnn states and inter-event time embedding
        self.layer_dense_2 = nn.Linear(
            in_features=hidden_size * 2, out_features=hidden_size
        )

        # mlp layers
        self.module_list = nn.ModuleList(
            [
                nn.Linear(in_features=hidden_size, out_features=hidden_size)
                for _ in range(num_mlp_layers - 1)
            ]
        )

        self.layer_dense_3 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=num_event_types),
            nn.Softplus(),
        )

        self.params_eps = torch.finfo(
            torch.float32
        ).eps  # ensure positiveness of parameters

        self.init_weights_positive()

    def init_weights_positive(self) -> None:
        """Initialize all weights to be positive."""
        for p in self.parameters():
            p.data = torch.abs(p.data)
            p.data = torch.clamp(p.data, min=self.params_eps)

    def forward(
        self, hidden_states: torch.Tensor, time_delta_seqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute cumulative hazard function and its derivative.
        Args:
            hidden_states: [batch_size, seq_len, (num_samples), hidden_size], hidden states at event times.
            time_delta_seqs: [batch_size, seq_len, (num_samples)], inter-event time seqs.
        Returns:
            tuple: cumulative hazard function values and their derivatives w.r.t. time deltas.
        """
        
        # Enable gradient computation specifically for the derivative calculation
        with torch.enable_grad():
            
            for p in self.parameters():
                p.data = torch.clamp(p.data, min=self.params_eps)

            time_delta_seqs.requires_grad_(True)

            # [batch_size, seq_len, (num_samples), hidden_size] or [batch_size, 1, (num_samples), hidden_size] when compute_last_step_only is True
            t = self.layer_dense_1(time_delta_seqs.unsqueeze(dim=-1))

            # [batch_size, seq_len, (num_samples), hidden_size] or [batch_size, 1, (num_samples), hidden_size] when compute_last_step_only is True
            if len(t.shape) == 4:
                t = t.expand(
                    -1,
                    hidden_states.shape[1],
                    -1,
                    -1,
                )  # expand seq_len dimension when compute_last_step_only is True

            out = torch.tanh(self.layer_dense_2(torch.cat([hidden_states, t], dim=-1)))
            for layer in self.module_list:
                out = torch.tanh(layer(out))

            # [batch_size, seq_len, (num_samples), num_event_types]
            integral_lambda = self.layer_dense_3(out)

            # [batch_size, seq_len, (num_samples), num_event_types]
            if self.proper_marked_intensities:
                derivative_integral_lambdas = []
                for i in range(integral_lambda.shape[-1]):  # iterate over marks
                    derivative_integral_lambdas.append(
                        grad(
                            integral_lambda[..., i].mean(),
                            time_delta_seqs,
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                    )
                derivative_integral_lambda = torch.stack(
                    derivative_integral_lambdas, dim=-1
                )  # TODO: Check that it is okay to iterate over marks like this
            else:
                derivative_integral_lambda = grad(
                    integral_lambda.sum(dim=-1).mean(),
                    time_delta_seqs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                derivative_integral_lambda = (
                    derivative_integral_lambda.unsqueeze(-1).expand(
                        *derivative_integral_lambda.shape, self.num_event_types
                    )
                    / self.num_event_types
                )

        return integral_lambda, derivative_integral_lambda


class FullyNN(NeuralModel):
    """Torch implementation of
    Fully Neural Network based Model for General Temporal Point Processes, NeurIPS 2019.
    https://arxiv.org/abs/1905.09690

    ref: https://github.com/KanghoonYoon/torch-neuralpointprocess/blob/master/module.py;
        https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(
        self,
        *,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        **kwargs,
    ):
        """Initialize the model

        Args:
            model_config (new_ltpp.ModelConfig): config of model specs.
        """
        super(FullyNN, self).__init__(**kwargs)

        self.hidden_size = self.hidden_size
        self.rnn_type = rnn_type
        self.rnn_list = [nn.LSTM, nn.RNN, nn.GRU]
        self.n_layers = num_layers

        # Initialize type embedding layer
        self.layer_type_emb = nn.Embedding(
            num_embeddings=self.num_event_types + 1,  # +1 for pad token
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_token_id,
        )

        for sub_rnn_class in self.rnn_list:
            if sub_rnn_class.__name__ == self.rnn_type:
                self.layer_rnn = sub_rnn_class(
                    input_size=1 + self.hidden_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.n_layers,
                    batch_first=True,
                    # dropout=self.specs["dropout"],
                )

        self.layer_intensity = CumulHazardFunctionNetwork(
            hidden_size=self.hidden_size,
            num_event_types=self.num_event_types,
        )

    def forward(
        self,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
    ) -> torch.Tensor:
        """Call the model

        Args:
            time_seqs: [batch_size, seq_len], timestamp seqs.
            time_delta_seqs: [batch_size, seq_len], inter-event time seqs.
            type_seqs: [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        type_embedding = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size + 1]
        rnn_input = torch.cat((type_embedding, time_delta_seqs.unsqueeze(-1)), dim=-1)

        # [batch_size, seq_len, hidden_size]
        # states right after the event
        hidden_states, _ = self.layer_rnn(rnn_input)

        return hidden_states

    def loglike_loss(self, batch: Batch) -> Tuple[torch.Tensor, int]:
        """Compute the loglike loss.

        Args:
            batch: batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        # [batch_size, seq_len]
        time_delta_seqs = batch.time_delta_seqs
        type_seqs = batch.type_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_delta_seqs[:, :-1],
            type_seqs[:, :-1],
        )
        # [batch_size, seq_len, num_event_types]
        integral_lambda, derivative_integral_lambda = self.layer_intensity.forward(
            hidden_states, time_delta_seqs[:, 1:]
        )

        # First, add an epsilon to every marked intensity for stability
        derivative_integral_lambda += self.eps

        # Compute components for each LL term
        log_marked_event_lambdas = derivative_integral_lambda.log()

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(
                0, 2, 1
            ),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seqs[:, 1:],
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction="none",  # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # [batch_size, seq_len]
        # multiplied by sequence mask
        non_event_ll = integral_lambda.sum(-1) * batch_non_pad_mask[:, 1:]
        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        loss = -(event_ll - non_event_ll).sum()

        return loss, num_events

    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute hidden states at sampled delta times.

        Args:
            time_seqs: [batch_size, seq_len], times seqs.
            time_delta_seqs: [batch_size, seq_len], time delta seqs.
            type_seqs: [batch_size, seq_len], event type seqs.
            sample_dtimes: [batch_size, seq_len, num_samples], sampled delta times.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled delta times.
        """

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
        )

        num_samples = sample_dtimes.size()[-1]
        batch_size, seq_len, hidden_size = hidden_states.shape

        hidden_states_ = hidden_states[..., None, :].expand(
            batch_size, seq_len, num_samples, hidden_size
        )
        _, derivative_integral_lambda = self.layer_intensity.forward(
            hidden_states=hidden_states_, # [batch_size, seq_len, num_samples, hidden_size]
            time_delta_seqs=sample_dtimes, # [batch_size, seq_len, num_samples] or 
        )

        if compute_last_step_only:
            lambdas = derivative_integral_lambda[:, -1:, :, :]
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = derivative_integral_lambda
        return lambdas
