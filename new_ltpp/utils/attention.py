"""Helpers for on-the-fly attention masks."""

from __future__ import annotations

import torch


def get_causal_attn_mask(
    size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Prepare input and the causal attention mask for a PyTorch multi-head attention layer.

    Args:
        input (torch.Tensor): The input tensor of shape (batch_size, seq_len, feature_dim).
        device (torch.device | str): The device where tensors should be placed.

    Returns:
        torch.Tensor: The causal attention mask.
    """
    # Apply causal masking

    # Create the causal mask
    attn_mask = torch.triu(
        torch.ones((size, size), device=device) * float("-inf"), diagonal=1
    )

    return attn_mask
