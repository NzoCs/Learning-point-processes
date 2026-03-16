import torch


def _forward_fill_padding(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace padded positions with the last valid embedding value.

    A constant path after the last event produces zero increments,
    so the signature kernel naturally ignores it.

    Args:
        emb: Embedding tensor (B, L', C) where L' may be L or 2*L for constant_interpolant.
        mask: Valid event mask (B, L) with True for real events.

    Returns:
        Embedding with padded positions forward-filled. Same shape as input.
    """
    B, L_emb, C = emb.shape
    B_mask, L_mask = mask.shape

    # For constant_interpolant, L_emb = 2 * L_mask: expand mask to match
    if L_emb == 2 * L_mask:
        # Each event i maps to positions 2*i and 2*i+1
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 2).reshape(B, L_emb)
    else:
        expanded_mask = mask

    # Find the index of the last valid position per sequence
    # last_valid_idx: (B,) — index of the last True in the mask
    last_valid_idx = expanded_mask.long().cumsum(dim=1).argmax(dim=1)  # (B,)

    # Gather the last valid embedding for each sequence
    last_valid_emb = emb[torch.arange(B, device=emb.device), last_valid_idx]  # (B, C)

    # Expand to full sequence length and fill where mask is False
    fill_value = last_valid_emb.unsqueeze(1).expand_as(emb)  # (B, L', C)
    emb = torch.where(expanded_mask.unsqueeze(-1), emb, fill_value)

    return emb


def _get_embedding(
    num_discretization_points: int,
    embedding_type: str,
    num_event_types: int,
    time_seqs: torch.Tensor,
    type_seqs: torch.Tensor,
) -> torch.Tensor:
    """Get the embedding of the sequences. Concatenate ponctual point process and time counting process.
    Args:
        embedding_type: Type of embedding to use
        num_event_types: Number of event types
        time_seqs: Batch of sequences of shape (B, L)
        type_seqs: Batch of type sequences of shape (B, L) with integer type indices
    Returns:
        torch.Tensor: (B, L, 2 + num_event_types) or (B, 2*L, 2 + num_event_types).
    """

    # Ne gère pas encore les masques à implementer !

    match embedding_type:
        case "linear_interpolant":
            # NOTE: No normalization here - should be done globally in compute_gram_matrix
            # to preserve relative time scales between sequences
            normalized_time_seqs = time_seqs  # (B, L)

            # Normalize counting sequences to [0, 1] range
            counting_seqs = torch.arange(
                time_seqs.shape[1], device=time_seqs.device, dtype=time_seqs.dtype
            ).expand(time_seqs.shape[0], -1)  # (B, L)
            normalized_counting_seqs = counting_seqs / (
                time_seqs.shape[1] - 1 + 1e-8
            )  # (B, L)

            # Create one-hot encoding for types
            # type_seqs is (B, L) with integer indices
            type_one_hot = torch.nn.functional.one_hot(
                type_seqs.long(), num_classes=num_event_types
            ).to(time_seqs.dtype)  # (B, L, num_event_types)

            return torch.cat(
                [
                    normalized_time_seqs.unsqueeze(-1),
                    normalized_counting_seqs.unsqueeze(-1),
                    type_one_hot,
                ],
                dim=-1,
            )  # (B, L, 2 + num_event_types)
        case "constant_interpolant":
            # NOTE: No normalization here - should be done globally in compute_gram_matrix
            # to preserve relative time scales between sequences
            normalized_time_seqs = time_seqs  # (B, L)

            # create the sequence of [t_1, t_1, t_2, t_2, ..., t_n, t_n]
            pair_indexes = (
                torch.arange(time_seqs.shape[1], device=time_seqs.device).expand(
                    time_seqs.shape[0], -1
                )
                * 2
            )
            out_time_seqs = torch.zeros(
                (time_seqs.shape[0], time_seqs.shape[1] * 2),
                dtype=time_seqs.dtype,
                device=time_seqs.device,
            )  # (B, 2*L)
            out_time_seqs[:, pair_indexes] = normalized_time_seqs
            out_time_seqs[:, pair_indexes + 1] = normalized_time_seqs

            # Normalize counting sequences
            counting_seqs = torch.arange(
                time_seqs.shape[1], dtype=time_seqs.dtype, device=time_seqs.device
            ).expand(time_seqs.shape[0], -1)
            normalized_counting_seqs = counting_seqs / (time_seqs.shape[1] - 1 + 1e-8)

            padded_counting_seqs = torch.zeros(
                (time_seqs.shape[0], time_seqs.shape[1] * 2),
                dtype=time_seqs.dtype,
                device=time_seqs.device,
            )
            padded_counting_seqs[:, pair_indexes] = normalized_counting_seqs
            padded_counting_seqs[:, pair_indexes + 1] = normalized_counting_seqs

            # Create one-hot encoding for types and duplicate
            type_one_hot = torch.nn.functional.one_hot(
                type_seqs.long(), num_classes=num_event_types
            ).to(time_seqs.dtype)  # (B, L, num_event_types)

            # Duplicate one-hot vectors for constant interpolant
            out_type_one_hot = torch.zeros(
                (time_seqs.shape[0], time_seqs.shape[1] * 2, num_event_types),
                dtype=time_seqs.dtype,
                device=time_seqs.device,
            )  # (B, 2*L, num_event_types)
            out_type_one_hot[:, pair_indexes] = type_one_hot
            out_type_one_hot[:, pair_indexes + 1] = type_one_hot

            return torch.cat(
                [
                    out_time_seqs.unsqueeze(-1),
                    padded_counting_seqs.unsqueeze(-1),
                    out_type_one_hot,
                ],
                dim=-1,
            )  # (B, 2*L, 2 + num_event_types)
        case _:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
