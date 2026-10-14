"""Invert an integrated intensity, including its atom at infinite waiting time."""

import torch


@torch.no_grad()
def invert_hazard(hazard, limit, num_samples, *, conditional=False, max_rounds=10000):
    """Return independent delays; infinity means that no next event exists.

    `limit` is the analytic integral to infinity, never a numerical tail cutoff.
    Conditional draws are explicitly conditioned on existence of the next event.
    """
    if torch.isnan(limit).any() or (limit < 0).any():
        raise ValueError("Invalid total integrated intensity")
    target = torch.empty(
        *limit.shape, num_samples, device=limit.device, dtype=limit.dtype
    ).exponential_()
    if conditional:
        # Exp(1) truncated at the total hazard. Stable even for a tiny limit.
        truncated = -torch.logaddexp(
            -target, torch.log(-torch.expm1(-target)) - limit.unsqueeze(-1)
        )
        target = torch.where(torch.isfinite(limit).unsqueeze(-1), truncated, target)
    finite = (target < limit.unsqueeze(-1)) & (limit.unsqueeze(-1) > 0)
    low = torch.zeros_like(target)
    high = torch.ones_like(target)
    rounds = 0
    while True:
        below = finite & (hazard(high) < target)
        if not below.any():
            break
        if rounds >= max_rounds or not torch.isfinite(high[below] * 2).all():
            raise RuntimeError(
                "Hazard inversion could not bracket finite next events safely"
            )
        high = torch.where(below, high * 2, high)
        rounds += 1
    # Floating point bisection, without a waiting-time cap.
    for _ in range(64):
        mid = low + (high - low) / 2
        below = hazard(mid) < target
        low = torch.where(finite & below, mid, low)
        high = torch.where(finite & ~below, mid, high)
    return torch.where(finite, high, torch.full_like(high, torch.inf))
