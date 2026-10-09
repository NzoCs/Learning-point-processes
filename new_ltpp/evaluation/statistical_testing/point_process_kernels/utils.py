import torch

SIGNATURE_PATH_PREPARATION = "masked_valid_count_v2"
SIGNATURE_PATH_REPRESENTATION = "counting_grid"


def _get_embedding(
    num_discretization_points: int,
    embedding_type: str,
    num_event_types: int,
    time_seqs: torch.Tensor,
    type_seqs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Get the embedding of the sequences evaluated on a regular time grid.
       This replaces the exact jump fitting with a discretized counting process.
       Returns the multi-dimensional counting process (t, N(t), N_1(t), ..., N_k(t)).
    Args:
        num_discretization_points: Number of points for the time grid
        embedding_type: counting_grid; linear/constant are legacy aliases.
        num_event_types: Number of event types
        time_seqs: Batch of sequences of shape (B, L) normalized to [0, 1]
        type_seqs: Batch of type sequences of shape (B, L) with integer type indices
        mask: Valid event mask (B, L) with True for real events.
    Returns:
        torch.Tensor: (B, num_discretization_points, 2 + num_event_types).
    """
    if embedding_type not in ("counting_grid", "linear", "constant"):
        raise ValueError(f"Unsupported signature path representation: {embedding_type}")
    B, L = time_seqs.shape
    device = time_seqs.device
    dtype = time_seqs.dtype

    if mask is None:
        mask = torch.ones_like(time_seqs, dtype=torch.bool)

    # 1) Create regular time grid
    time_grid = torch.linspace(
        0.0, 1.0, num_discretization_points, device=device, dtype=dtype
    )
    time_grid_exp = time_grid.unsqueeze(0).expand(B, -1)  # (B, D)

    # Sort a local representation only. Keeping marks and masks aligned also
    # handles left/interspersed padding and preserves the order of tied events.
    # No input tensor is mutated; infinity is used only for searchsorted.
    time_seqs_inf, order = time_seqs.masked_fill(~mask, float("inf")).sort(
        dim=1, stable=True
    )
    sorted_types = type_seqs.gather(1, order)
    sorted_mask = mask.gather(1, order)

    # idx is the number of valid events <= t
    idx = torch.searchsorted(time_seqs_inf, time_grid_exp.contiguous(), side="right")

    # 3) Compute cumulative counts per event type
    # Clamp to [0, num_event_types - 1] to prevent one_hot out-of-bound crashes on padding tokens.
    # The padded elements will be zeroed out when multiplying by the mask.
    type_seqs_clamped = torch.clamp(sorted_types.long(), min=0, max=num_event_types - 1)
    one_hot = torch.nn.functional.one_hot(
        type_seqs_clamped, num_classes=num_event_types
    ).to(dtype)
    one_hot = one_hot * sorted_mask.unsqueeze(-1).to(dtype)
    cum_counts = torch.cumsum(one_hot, dim=1)  # (B, L, num_event_types)

    # Prepend zeros because idx=0 means 0 events have occurred
    zeros = torch.zeros((B, 1, num_event_types), device=device, dtype=dtype)
    cum_counts_padded = torch.cat(
        [zeros, cum_counts], dim=1
    )  # (B, L+1, num_event_types)

    # Gather the type counts at each grid point
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, num_event_types)
    type_counting_seqs = torch.gather(
        cum_counts_padded, 1, idx_expanded
    )  # (B, D, num_event_types)

    # Overall counting sequence is just idx
    counting_seqs = idx.to(dtype)

    # Preserve the historical n-1 scaling for unpadded paths with n >= 2,
    # using each path's actual event count rather than the padded batch width.
    # Empty/single-event paths use a unit denominator and remain finite.
    denominator = (mask.sum(dim=1) - 1).clamp_min(1).to(dtype) + 1e-8
    normalized_counting_seqs = counting_seqs / denominator.unsqueeze(1)

    # Concatenate time_grid, counting sequence and type counting sequences
    return torch.cat(
        [
            time_grid_exp.unsqueeze(-1),
            normalized_counting_seqs.unsqueeze(-1),
            type_counting_seqs,
        ],
        dim=-1,
    )  # (B, num_discretization_points, 2 + num_event_types)
