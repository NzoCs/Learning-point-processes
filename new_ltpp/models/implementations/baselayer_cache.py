import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Cache type aliases — pour la lisibilité
KVCache = Tuple[torch.Tensor, torch.Tensor]  # (K, V) — [B, L_past, D]
LayerCache = Optional[KVCache]  # cache d'une couche
ModelCache = Optional[List[LayerCache]]  # cache de tout le modèle


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION SCALÉE PERSONNALISÉE
# ─────────────────────────────────────────────────────────────────────────────


class ScaledSoftplus(nn.Module):
    """Softplus with learnable per-mark scale β — no cache needed."""

    def __init__(self, num_marks: int, threshold: float = 20.0):
        super().__init__()
        self.threshold = threshold
        self.log_beta = nn.Parameter(torch.zeros(num_marks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.log_beta.exp()  # [num_marks]
        beta_x = beta * x
        return torch.where(
            beta_x <= self.threshold,
            F.softplus(beta_x) / beta,
            x,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-HEAD ATTENTION avec KV cache
# ─────────────────────────────────────────────────────────────────────────────


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention avec KV cache pour inférence incrémentale.

    Convention de cache
    -------------------
    kv_cache : Optional[Tuple[Tensor, Tensor]]
        (K_past, V_past) de shape [B, L_past, D]
        None  → pas de cache (training ou premier step)

    Retour
    ------
    Toujours (output, kv_cache_new) où kv_cache_new contient
    [K_past | K_new], [V_past | V_new] — prêt pour le step suivant.
    """

    def __init__(
        self,
        n_head: int,
        d_input: int,
        d_model: int,
        dropout: float = 0.1,
        output_linear: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model doit être divisible par n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.input_proj = (
            nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()
        )

        # Projections Q, K, V séparées — nécessaire pour injecter le cache sur K/V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model) if output_linear else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] → [B, n_head, L, d_head]"""
        B, L, _ = x.shape
        return x.view(B, L, self.n_head, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_head, L, d_head] → [B, L, D]"""
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(
        self,
        query: torch.Tensor,  # [B, L_q, D_in]
        key: torch.Tensor,  # [B, L_k, D_in]
        value: torch.Tensor,  # [B, L_v, D_in]
        attn_mask: Optional[torch.Tensor] = None,  # [L_q, L_total] ou [B, L_q, L_total]
        kv_cache: LayerCache = None,  # (K_past, V_past) | None
        output_weight: bool = False,
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Returns
        -------
        out       : [B, L_q, D]
        kv_cache  : (K_full, V_full) — [B, L_past + L_k, D]
        attn_w    : [B, n_head, L_q, L_total]  (seulement si output_weight=True)
        """
        q = self.input_proj(query)
        k = self.input_proj(key)
        v = self.input_proj(value)

        Q = self._split_heads(self.W_q(q))  # [B, H, L_q, d_head]
        K = self._split_heads(self.W_k(k))  # [B, H, L_k, d_head]
        V = self._split_heads(self.W_v(v))  # [B, H, L_k, d_head]

        # ── Append du cache ──
        if kv_cache is not None:
            K_past, V_past = kv_cache
            # K_past: [B, H, L_past, d_head]
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)

        # Nouveau cache = K/V complets (avant la tête de sortie)
        new_kv_cache: KVCache = (K, V)

        # ── Scaled dot-product attention ──
        # Flash attention si dispo (PyTorch ≥ 2.0), sinon fallback natif
        scale = math.sqrt(self.d_head)

        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L_q, L_total]

        attn_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / scale
        )  # [B, H, L_q, L_total]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # additive mask (−inf pour causal)

        attn_w = F.softmax(attn_scores, dim=-1)
        attn_w = self.dropout(attn_w)

        out = self._merge_heads(torch.matmul(attn_w, V))  # [B, L_q, D]
        out = self.W_o(out)
        out = self.out_proj(out)

        if output_weight:
            return out, new_kv_cache, attn_w
        return out, new_kv_cache


# ─────────────────────────────────────────────────────────────────────────────
# SUBLAYER CONNECTION (résidu + norm) — cache-transparent
# ─────────────────────────────────────────────────────────────────────────────


class SublayerConnection(nn.Module):
    """
    x → LayerNorm → sublayer → résidu + dropout.

    Le sublayer peut retourner :
      - un Tensor seul  (ex. FFN)
      - (Tensor, cache) (ex. Attention)

    Dans les deux cas, SublayerConnection retourne (x_out, cache).
    cache est None si le sublayer n'en produit pas.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable,
    ) -> Tuple[torch.Tensor, LayerCache]:
        result = sublayer(self.norm(x))

        if isinstance(result, tuple):
            out, cache = result[0], result[1]  # (Tensor, KVCache)
        else:
            out, cache = result, None

        return x + self.dropout(out), cache


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER LAYER — cache par couche
# ─────────────────────────────────────────────────────────────────────────────


class EncoderLayer(nn.Module):
    """
    Une couche Transformer : self-attention + FFN optionnel.

    Cache
    -----
    Accepte layer_cache : LayerCache = (K_past, V_past) | None
    Retourne toujours   (output, layer_cache_new)

    Note : le cache ne concerne que l'attention (le FFN est stateless).
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        feed_forward: Optional[nn.Module] = None,
        use_residual: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual

        if use_residual:
            self.sublayer_attn = SublayerConnection(d_model, dropout)
            self.sublayer_ffn = (
                SublayerConnection(d_model, dropout) if feed_forward else None
            )

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        attn_mask: Optional[torch.Tensor] = None,
        layer_cache: LayerCache = None,  # (K_past, V_past) | None
    ) -> Tuple[torch.Tensor, LayerCache]:
        """
        Returns
        -------
        x_out      : [B, L, D]
        layer_cache: (K_full, V_full) mis à jour
        """
        if self.use_residual:
            # SublayerConnection gère le résidu — on lui passe l'attention comme sublayer
            x, new_cache = self.sublayer_attn(
                x,
                lambda x_norm: self.self_attn(
                    x_norm,
                    x_norm,
                    x_norm,
                    attn_mask=attn_mask,
                    kv_cache=layer_cache,
                ),
            )
            if self.feed_forward is not None and self.sublayer_ffn is not None:
                x, _ = self.sublayer_ffn(x, self.feed_forward)  # FFN — pas de cache
        else:
            out, new_cache = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                kv_cache=layer_cache,
            )
            x = out
            if self.feed_forward is not None:
                x = self.feed_forward(x)

        return x, new_cache


# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE TEMPOREL — pas de cache (stateless)
# ─────────────────────────────────────────────────────────────────────────────


class TimePositionalEncoding(nn.Module):
    """Encodage sinusoïdal basé sur le temps absolu."""

    div_term: torch.Tensor

    def __init__(self, d_model: int, device: str | torch.device, max_len: int = 5000):
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
    """Encodage sinusoïdal décalé par les inter-arrivées (SAHP)."""

    position: torch.Tensor
    div_term: torch.Tensor

    def __init__(
        self, d_model: int, max_len: int = 5000, device: str | torch.device = "cpu"
    ):
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


# ─────────────────────────────────────────────────────────────────────────────
# DNN — stateless, pas de cache
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — gestion du ModelCache au niveau du modèle complet
# ─────────────────────────────────────────────────────────────────────────────


def init_model_cache(n_layers: int) -> ModelCache:
    """Crée un cache vide pour un modèle à n_layers couches."""
    return [None] * n_layers


def update_model_cache(
    cache: ModelCache,
    layer_idx: int,
    new_kv: KVCache,
) -> ModelCache:
    """Retourne un nouveau ModelCache avec la couche layer_idx mise à jour."""
    updated = list(cache)
    updated[layer_idx] = new_kv
    return updated
