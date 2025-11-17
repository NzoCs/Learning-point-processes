"""Helpers for on-the-fly attention masks."""

from __future__ import annotations

import torch


def _get_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )


def build_attention_mask_from_seq_mask(seq_non_pad_mask: torch.Tensor) -> torch.Tensor:
    """Create a causal attention mask that respects padding."""
    if seq_non_pad_mask.dim() != 2:
        raise ValueError("seq_non_pad_mask must be 2-dimensional")

    batch_size, seq_len = seq_non_pad_mask.shape
    device = seq_non_pad_mask.device

    causal = _get_causal_mask(seq_len, device)
    causal = causal.unsqueeze(0).expand(batch_size, -1, -1)

    padding = (~seq_non_pad_mask).unsqueeze(1).expand(-1, seq_len, -1)

    return causal | padding