from abc import ABC

import torch.nn as nn
from .basemodel import Model


class NeuralModel(Model, ABC):
    """
    Neural Temporal Point Process model.
    Inherits from Model.
    """

    def __init__(
        self,
        model_config,
        *,
        num_event_types: int,
        dtime_max: float,
        dropout: float,
        hidden_size: int,
        **kwargs,
    ):
        """
        Initialize the NeuralModel.

        Args:
            model_config: Configuration of the model.
            num_event_types: Number of event types.
        """
        super(NeuralModel, self).__init__(
            model_config,
            num_event_types=num_event_types,
            dtime_max=dtime_max
        )

        self.num_event_types = num_event_types
        self.dropout = dropout
        self.hidden_size = hidden_size

        # Initialize type embedding
        self.layer_type_emb = nn.Embedding(
            num_embeddings=self.num_event_types + 1,  # have padding
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_token_id,
            device=self.device,
        )
