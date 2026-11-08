"""Explicit fixed-window observations; never infer exposure from the last event."""

import math

import torch

from new_ltpp.shared_types import Batch


def configurations(batch: Batch, horizon: float, num_marks: int):
    if not math.isfinite(horizon) or horizon <= 0 or num_marks < 1:
        raise ValueError("Positive finite horizon and mark count required")
    if batch.time_seqs.ndim != 2 or batch.time_seqs.shape != batch.type_seqs.shape:
        raise ValueError("Matching two-dimensional times and marks required")
    if batch.valid_event_mask.shape != batch.time_seqs.shape:
        raise ValueError("Mask shape mismatch")
    if batch.valid_event_mask.dtype != torch.bool or not len(batch.time_seqs):
        raise ValueError("Nonempty batch and boolean event mask required")
    rows = []
    for t, m, mask in zip(batch.time_seqs, batch.type_seqs, batch.valid_event_mask):
        t, m = t[mask].double(), m[mask]
        if not torch.isfinite(t).all() or not ((t > 0) & (t < horizon)).all():
            raise ValueError(
                "Events must lie strictly inside the fixed window; mask BOS"
            )
        if not (torch.diff(t) > 0).all():
            raise ValueError("Simple, strictly ordered event times required")
        if (
            not torch.isfinite(m).all()
            or not ((m == m.long()) & (m >= 0) & (m < num_marks)).all()
        ):
            raise ValueError("Invalid event mark")
        rows.append((t, m.long()))
    return rows


def from_configurations(rows, *, device="cpu"):
    if not rows:
        raise ValueError("Nonempty batch required")
    width = max(1, max(len(t) for t, _ in rows))
    times = torch.zeros(len(rows), width, dtype=torch.float64, device=device)
    marks = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    mask = torch.zeros_like(marks, dtype=torch.bool)
    for i, (t, m) in enumerate(rows):
        times[i, : len(t)] = torch.as_tensor(t, device=device)
        marks[i, : len(t)] = torch.as_tensor(m, device=device)
        mask[i, : len(t)] = True
    gaps = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
    return Batch(times, gaps.masked_fill(~mask, 0), marks, mask)
