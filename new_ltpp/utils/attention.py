"""Helpers for on-the-fly attention masks."""

from __future__ import annotations

import torch


def get_causal_attn_mask(
    size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Prépare l'entrée et le masque d'attention causal pour une couche d'attention multi-tête PyTorch.

    Args:
        input (torch.Tensor): Le tenseur d'entrée de forme (batch_size, seq_len, feature_dim).
        device (torch.device | str): Le dispositif sur lequel les tenseurs doivent être placés.

    Returns:
        torch.Tensor: Le masque d'attention causal.
    """
    # Appliquer le masque de non-remboursement

    # Créer le masque causal
    attn_mask = torch.triu(
        torch.ones((size, size), device=device) * float("-inf"), diagonal=1
    )

    return attn_mask
