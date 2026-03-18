import torch


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
        embedding_type: Kept for signature compatibility but essentially ignored
        num_event_types: Number of event types
        time_seqs: Batch of sequences of shape (B, L) normalized to [0, 1]
        type_seqs: Batch of type sequences of shape (B, L) with integer type indices
        mask: Valid event mask (B, L) with True for real events.
    Returns:
        torch.Tensor: (B, num_discretization_points, 2 + num_event_types).
    """
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

    # 2) Mask out padded times with inf so searchsorted disregards them
    time_seqs_inf = time_seqs.masked_fill(~mask, float("inf"))

    # idx is the number of valid events <= t
    idx = torch.searchsorted(time_seqs_inf, time_grid_exp, side="right")

    # 3) Compute cumulative counts per event type
    one_hot = torch.nn.functional.one_hot(
        type_seqs.long(), num_classes=num_event_types
    ).to(dtype)
    one_hot = one_hot * mask.unsqueeze(-1).to(dtype)
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

    # We normalized counting sequences before by max steps, let's normalize by overall steps if needed,
    # but using raw counts for a step process is standard. Since previous code used max possible length `L`:
    # But usually not normalizing counts is fine for sigkernel, since space kernel applies scaling.
    # Previous code: normalized_counting_seqs = counting_seqs / (time_seqs.shape[1] - 1 + 1e-8)
    normalized_counting_seqs = counting_seqs / (L - 1 + 1e-8)

    # Concatenate time_grid, counting sequence and type counting sequences
    return torch.cat(
        [
            time_grid_exp.unsqueeze(-1),
            normalized_counting_seqs.unsqueeze(-1),
            type_counting_seqs,
        ],
        dim=-1,
    )  # (B, num_discretization_points, 2 + num_event_types)
