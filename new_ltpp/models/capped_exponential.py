"""Differentiable compensator for exp(min(a + w*t, log(cap)))."""

import math

import torch


def capped_exponential_integral(a, w, t, cap=1e5):
    """Elementwise integral from zero to nonnegative t, including w == 0.

    Inputs broadcast per history, query and mark. The local series preserves
    dH/dw = exp(a)*t**2/2 at zero slope below the cap.
    """
    log_cap = math.log(cap)
    safe_w = torch.where(w == 0, torch.ones_like(w), w)
    crossing = ((log_cap - a) / safe_w).clamp_min(0)
    duration = torch.where(
        w > 0,
        torch.minimum(t, crossing),
        torch.where(a > log_cap, (t - crossing).clamp_min(0), t),
    )
    duration = torch.where(
        w == 0, torch.where(a > log_cap, torch.zeros_like(t), t), duration
    )
    z = w * duration
    small = z.abs() < 1e-4
    safe_z = torch.where(small, torch.ones_like(z), z)
    # Avoid overflow in an inactive branch of torch.where as well.
    ordinary = torch.expm1(torch.where(small, torch.zeros_like(z), z)) / safe_z
    series_z = torch.where(small, z, torch.zeros_like(z))
    series = (
        1
        + series_z / 2
        + series_z.square() / 6
        + series_z.pow(3) / 24
        + series_z.pow(4) / 120
    )
    ratio = torch.where(small, series, ordinary)
    return a.clamp(max=log_cap).exp() * duration * ratio + cap * (t - duration)


def capped_exponential_limit(a, w, cap=1e5):
    """Infinite-time integral; nonnegative slopes have infinite mass."""
    negative = w < 0
    safe_w = torch.where(negative, w, -torch.ones_like(w))
    plateau = ((a - math.log(cap)) / (-safe_w)).clamp_min(0)
    finite = a.clamp(max=math.log(cap)).exp() / (-safe_w) + cap * plateau
    return torch.where(negative, finite, torch.full_like(finite, torch.inf))
