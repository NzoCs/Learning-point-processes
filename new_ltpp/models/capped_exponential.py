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
    start = a.clamp(max=log_cap)
    rate = start.exp()
    # For growing intensities, exp(start)*expm1(z) may be 0*infinity even
    # though start+z is capped and the integral is finite. Subtract endpoint
    # rates instead, and keep every inactive torch.where branch finite.
    end = (start + z.clamp_min(0)).clamp(max=log_cap)
    growing_integral = (end.exp() - rate) / safe_w
    decreasing_integral = rate * torch.expm1(z.clamp(max=0)) / safe_w
    series_z = torch.where(small, z, torch.zeros_like(z))
    series = (
        1
        + series_z / 2
        + series_z.square() / 6
        + series_z.pow(3) / 24
        + series_z.pow(4) / 120
    )
    ordinary = torch.where(z > 0, growing_integral, decreasing_integral)
    integral = torch.where(small, rate * duration * series, ordinary)
    return integral + cap * (t - duration)


def capped_exponential_limit(a, w, cap=1e5):
    """Infinite-time integral; nonnegative slopes have infinite mass."""
    negative = w < 0
    safe_w = torch.where(negative, w, -torch.ones_like(w))
    plateau = ((a - math.log(cap)) / (-safe_w)).clamp_min(0)
    finite = a.clamp(max=math.log(cap)).exp() / (-safe_w) + cap * plateau
    return torch.where(negative, finite, torch.full_like(finite, torch.inf))
