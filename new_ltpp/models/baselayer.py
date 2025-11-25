import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn


# -------------------------------------------------------------------------
# ACTIVATION SCALÉE PERSONNALISÉE (pas dans PyTorch)
# -------------------------------------------------------------------------
class ScaledSoftplus(nn.Module):
    def __init__(self, num_marks: int, threshold: float = 20.0):
        super().__init__()
        self.threshold = threshold
        self.log_beta = nn.Parameter(torch.zeros(num_marks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.log_beta.exp()  # [num_marks]
        beta_x = beta * x
        return torch.where(
            beta_x <= self.threshold,
            torch.nn.functional.softplus(beta_x) / beta,
            x,
        )


# -------------------------------------------------------------------------
# PYTORCH NATIF POUR MULTI-HEAD ATTENTION
# -------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_input: int,
        d_model: int,
        dropout: float = 0.1,
        output_linear: bool = False,
    ):
        super().__init__()
        self.output_linear = output_linear

        # projection d’entrée si nécessaire
        self.input_proj = (
            nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()
        )

        # PyTorch native MHA — OP FUSED, OPTIMISÉ, FLASH-ATTENTION (si dispo)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,  # évite les permutes
        )

        self.out_proj = nn.Linear(d_model, d_model) if output_linear else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        output_weight: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        q = self.input_proj(query)
        k = self.input_proj(key)
        v = self.input_proj(value)

        attn_out, attn_weights = self.mha(
            q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        out = self.out_proj(attn_out)

        return (out, attn_weights) if output_weight else out


# -------------------------------------------------------------------------
# RÉSIDENTIEL + NORM (simplifié avec code PyTorch natif)
# -------------------------------------------------------------------------
class SublayerConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


# -------------------------------------------------------------------------
# ENCODER LAYER (self-attn + MLP)
# -------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attn: nn.Module,
        feed_forward: Optional[nn.Module] = None,
        use_residual: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual

        if use_residual:
            self.sublayers = nn.ModuleList(
                [
                    SublayerConnection(d_model, dropout),
                    SublayerConnection(d_model, dropout),
                ]
            )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.use_residual:
            x = self.sublayers[0](x, lambda x: self.self_attn(
                x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                ))
            if self.feed_forward is not None:
                x = self.sublayers[1](x, self.feed_forward)
            return x

        # Sans résiduel
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return self.feed_forward(x) if self.feed_forward is not None else x


# -------------------------------------------------------------------------
# ENCODAGE TEMPOREL THP / SAHP (à garder, pas dans Torch)
# -------------------------------------------------------------------------
class TimePositionalEncoding(nn.Module):

    div_term: torch.Tensor
    
    def __init__(self, d_model: int, max_len: int = 5000, device: str | torch.device = "cpu"):
        super().__init__()
        i = torch.arange(0, d_model, device=device)
        div_term = (2 * (i // 2) * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer("div_term", div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = x.unsqueeze(-1) * self.div_term
        result[..., 0::2] = torch.sin(result[..., 0::2])
        result[..., 1::2] = torch.cos(result[..., 1::2])
        return result


class TimeShiftedPositionalEncoding(nn.Module):

    position: torch.Tensor
    div_term: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000, device: str | torch.device = "cpu"):
        super().__init__()
        position = torch.arange(max_len, device=device).float().unsqueeze(1)
        div_term = torch.arange(0, d_model, 2, device=device).float()
        div_term = (div_term * -(math.log(10000.0) / d_model)).exp()

        self.layer_time_delta = nn.Linear(1, d_model // 2, bias=False)

        self.register_buffer("position", position)
        self.register_buffer("div_term", div_term)

    def forward(self, x: torch.Tensor, interval: torch.Tensor) -> torch.Tensor:
        phi = self.layer_time_delta(interval.unsqueeze(-1))
        L = x.size(1)
        arc = (self.position[:L] * self.div_term).unsqueeze(0)
        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        return torch.cat([pe_sin, pe_cos], dim=-1)


# -------------------------------------------------------------------------
# DNN (MLP modernisé)
# -------------------------------------------------------------------------
class DNN(nn.Module):
    def __init__(
        self,
        inputs_dim: int,
        hidden_size: List[int],
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        layers = []
        in_dim = inputs_dim

        for h in hidden_size:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))

            layers.append(getattr(nn, activation)())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
