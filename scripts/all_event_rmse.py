"""All forecast origins, including the terminal censored wait; no model changes.

CPU/CUDA float64 quadrature is checked by order refinement. The permutation
unit is an entire trajectory, never an individual dependent waiting time.
"""

from dataclasses import dataclass

import numpy as np
import torch

from scripts.report_rmse_validation import check_rows, quadrature, waiting_mean

PROTOCOL = "common_reference_all_event_rmse_v1"
ATOL = 1e-11
RTOL = 1e-8


@dataclass
class Origins:
    times: np.ndarray
    prefix: np.ndarray
    path: np.ndarray
    targets: np.ndarray
    primary: np.ndarray
    count: np.ndarray


def forecast_origins(rows):
    """Origin zero + every event before T; separate four-origin horizon control.

    Each target is min(next wait, T-origin), which is fully observed even for
    the last censored interval. No target or future count enters the forecast.
    An event exactly at T is a target but starts no zero-exposure forecast.
    """
    check_rows(rows, 1.0)
    times, prefixes, paths, targets, primary, count = [], [], [], [], [], []
    for i, row in enumerate(rows):
        if np.any(np.diff(row) <= 0):
            raise ValueError("Simple-process event times must be strictly increasing")
        event_times = np.concatenate(([0.0], row[row < 1.0]))
        event_prefix = np.arange(len(event_times))
        next_times = np.concatenate((row, [1.0]))[: len(event_times)]
        times.extend(event_times)
        prefixes.extend(event_prefix)
        paths.extend([i] * len(event_times))
        targets.extend(next_times - event_times)
        primary.extend([True] * len(event_times))
        count.append(len(event_times))
        for time in (0.0, 0.25, 0.5, 0.75):
            pos = int(np.searchsorted(row, time, side="right"))
            times.append(time)
            prefixes.append(pos)
            paths.append(i)
            targets.append(float(row[pos] - time) if pos < len(row) else 1 - time)
            primary.append(False)
    return Origins(
        np.asarray(times),
        np.asarray(prefixes),
        np.asarray(paths),
        np.asarray(targets),
        np.asarray(primary),
        np.asarray(count),
    )


def _means(ages, mask, caps, process, order):
    nodes, weights = quadrature(order)
    u = caps[:, None] * torch.as_tensor((nodes + 1) / 2, device=caps.device)
    mu = process.rate * (1 - process.branching)
    if process.kind == "exp":
        excitation = process.branching * (torch.exp(-process.beta * ages) * mask).sum(1)
        hazard = mu * u + excitation[:, None] * (-torch.expm1(-process.beta * u))
    elif process.kind == "power":
        base = 1 + process.beta * ages
        hazard = mu * u
        # Quadrature nodes are streamed to bound memory independently of order.
        for start in range(0, order, 16):
            delay = u[:, start : start + 16]
            extra = process.branching * (
                (base ** (1 - process.exponent))[:, :, None]
                - (base[:, :, None] + process.beta * delay[:, None, :])
                ** (1 - process.exponent)
            )
            hazard[:, start : start + 16] += (extra * mask[:, :, None]).sum(1)
    else:
        raise ValueError("Unknown predictor")
    return (
        torch.exp(-hazard) @ torch.as_tensor(weights / 2, device=caps.device)
    ) * caps


def forecasts(rows, process, origins, *, device="cpu", chunk=256):
    if (
        process.kind not in ("poisson", "exp", "power")
        or chunk < 1
        or not np.isfinite(
            [process.rate, process.branching, process.beta, process.exponent]
        ).all()
        or process.rate <= 0
        or not 0 <= process.branching < 1
        or process.beta <= 0
        or process.exponent <= 1
    ):
        raise ValueError("Invalid predictor or numerical chunk")
    if device not in ("cpu", "cuda") or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise RuntimeError("Requested numerical backend unavailable; no fallback")
    caps = 1 - origins.times
    if process.kind == "poisson":
        return -np.expm1(-process.rate * caps) / process.rate
    max_events = max(1, max(map(len, rows)))
    padded = np.zeros((len(rows), max_events))
    for i, row in enumerate(rows):
        padded[i, : len(row)] = row
    history = torch.as_tensor(padded, dtype=torch.float64, device=device)
    index = torch.arange(max_events, device=device)
    result = np.empty(len(caps))
    with torch.no_grad():
        for start in range(0, len(caps), chunk):
            end = min(start + chunk, len(caps))
            path = torch.as_tensor(origins.path[start:end], device=device)
            time = torch.as_tensor(origins.times[start:end], device=device)
            prefix = torch.as_tensor(origins.prefix[start:end], device=device)
            cap = torch.as_tensor(caps[start:end], device=device)
            mask = index[None, :] < prefix[:, None]
            ages = torch.where(mask, time[:, None] - history[path], 0.0)
            if (ages < 0).any():
                raise ValueError("Future events entered a forecast")
            previous = _means(ages, mask, cap, process, 32)
            values = _means(ages, mask, cap, process, 64)
            pending = ~torch.isclose(previous, values, rtol=RTOL, atol=ATOL)
            for order in (128, 256):
                if not pending.any():
                    break
                selected = torch.nonzero(pending).flatten()
                refined = _means(
                    ages[selected], mask[selected], cap[selected], process, order
                )
                close = torch.isclose(values[selected], refined, rtol=RTOL, atol=ATOL)
                values[selected] = refined
                pending[selected] = ~close
            if pending.any():
                # Independent adaptive CPU integration of only unresolved origins.
                for local in torch.nonzero(pending).flatten().cpu().tolist():
                    j = start + local
                    value = waiting_mean(
                        rows[origins.path[j]][: origins.prefix[j]],
                        origins.times[j],
                        process,
                        caps[j],
                        order=256,
                    )
                    if not np.isclose(
                        float(values[local]), value, rtol=RTOL, atol=ATOL
                    ):
                        raise ArithmeticError("All-event quadrature did not converge")
                    values[local] = value
            if (
                not torch.isfinite(values).all()
                or not ((values > 0) & (values <= cap * (1 + 1e-12))).all()
            ):
                raise ArithmeticError("Invalid restricted prediction")
            result[start:end] = values.cpu().numpy()
    return result


def all_event_losses(rows, reference, *, device="cpu", chunk=256, verify=False):
    origins = forecast_origins(rows)
    prediction = forecasts(rows, reference, origins, device=device, chunk=chunk)
    if verify:
        # Technical sample chosen by fixed indices, not by errors or power.
        positions = sorted(set([0, 1, len(prediction) // 2, len(prediction) - 1]))
        for j in positions:
            value = waiting_mean(
                rows[origins.path[j]][: origins.prefix[j]],
                origins.times[j],
                reference,
                1 - origins.times[j],
                order=256,
            )
            np.testing.assert_allclose(prediction[j], value, rtol=RTOL, atol=ATOL)
    error = ((origins.targets - prediction) / prediction) ** 2
    primary = origins.primary
    total = np.bincount(
        origins.path[primary], weights=error[primary], minlength=len(rows)
    )
    control = (
        np.bincount(
            origins.path[~primary], weights=error[~primary], minlength=len(rows)
        )
        / 4
    )
    if not np.isfinite(total).all() or not np.isfinite(control).all():
        raise ArithmeticError("Nonfinite trajectory loss")
    return dict(
        path_mse=total / origins.count,
        landmark_full_mse=control,
        origins_per_path=origins.count,
    )
