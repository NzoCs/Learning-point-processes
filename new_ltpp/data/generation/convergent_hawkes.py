"""Finite-start Hawkes cluster generator for the convergent power-kernel family.

All ancestors from the chosen start remain in the cluster simulation. This is
Poisson branching, not thinning, and is not exact stationary initialization.
"""

import math

import numpy as np


def kernel(times, *, p=None, b=10.0, branching=0.5):
    t = np.asarray(times, dtype=float)
    if (
        not np.isfinite([b, branching]).all()
        or b <= 0
        or not 0 <= branching < 1
        or (p is not None and (not math.isfinite(p) or p <= 1))
    ):
        raise ValueError("Invalid stable Hawkes kernel parameters")
    if not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Finite nonnegative lags required")
    return (
        branching * b * (np.exp(-b * t) if p is None else (1 + b * t / (p - 1)) ** (-p))
    )


def tail_mass(time, *, p=None, b=10.0, branching=0.5):
    if time < 0:
        raise ValueError("Nonnegative tail origin required")
    return branching * (
        math.exp(-b * time) if p is None else (1 + b * time / (p - 1)) ** (1 - p)
    )


def simulate(
    rng,
    *,
    burn,
    horizon=1.0,
    p=None,
    mu=50.0,
    b=10.0,
    branching=0.5,
    max_events=2_000_000,
):
    kernel([0.0], p=p, b=b, branching=branching)
    if (
        not np.isfinite([burn, horizon, mu]).all()
        or burn < 0
        or horizon <= 0
        or mu <= 0
        or max_events < 1
    ):
        raise ValueError("Invalid exposure, immigration rate or event budget")
    ancestors = rng.uniform(-burn, horizon, rng.poisson(mu * (burn + horizon)))
    generations, total = [ancestors], len(ancestors)
    while len(ancestors):
        if total > max_events:
            raise RuntimeError(
                "Event budget exceeded; no trajectory truncated or excluded"
            )
        parents = np.repeat(ancestors, rng.poisson(branching, len(ancestors)))
        exponentials = rng.exponential(size=len(parents))
        with np.errstate(over="raise", invalid="raise"):
            waits = (
                exponentials / b
                if p is None
                else (p - 1) / b * np.expm1(exponentials / (p - 1))
            )
        children = parents + waits
        ancestors = children[children < horizon]
        total += len(ancestors)
        generations.append(ancestors)
    full = np.sort(np.concatenate(generations))
    return full[(full > 0) & (full < horizon)], full
