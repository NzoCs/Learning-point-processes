"""Helpers for on-the-fly attention masks."""

from __future__ import annotations

import torch



def build_causal_attn_mask(
    len: int,
    device: torch.device | str,
    dtype=torch.bool,
) -> torch.Tensor:
    """
    Retourne un masque causal pour la MHA PyTorch :
    - shape (q_len, k_len)
    - True = MASQUÉ
    """
    # positions futures interdites → causal
    mask = torch.triu(
        torch.ones(len, len, dtype=dtype, device=device),
        diagonal=1
    )

    float_mask = mask.masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float("-inf"))

    return float_mask