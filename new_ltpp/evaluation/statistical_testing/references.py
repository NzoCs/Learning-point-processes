"""Frozen intensity references for fixed-window prediction and point-process Stein.

No model loading aliases or artifact conversion. Neural evaluation recomputes
the whole affected future after an insertion; training loss is not a density.
"""

import math
from functools import lru_cache
from typing import Protocol

import numpy as np
import torch


@lru_cache(maxsize=32)
def legendre_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


class IntensityReference(Protocol):
    num_event_types: int

    def intensities(self, times, marks, origin, delays): ...


class IntegratedIntensityReference:
    """Numerically integrated conditional density and restricted waiting mean.

    Subclasses provide causal marked intensities after a fixed history. The
    quadrature is deterministic and checked at doubled resolution; failures
    raise rather than clipping scores or dropping paths.
    """

    def __init__(self, *, order=32, rtol=2e-4, atol=1e-8):
        if order < 4 or not 0 < rtol < 1 or not 0 < atol < 1:
            raise ValueError("Invalid quadrature controls")
        self.order, self.rtol, self.atol = order, rtol, atol

    def _integrate(self, function, cap):
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("Positive finite exposure required")
        previous = None
        for order in [self.order, self.order * 2, self.order * 4]:
            z, w = legendre_rule(order)
            value = float(np.asarray(function((z + 1) * cap / 2)) @ w * cap / 2)
            if not math.isfinite(value):
                raise ValueError("Nonfinite intensity integral")
            if previous is not None and np.isclose(
                value, previous, rtol=self.rtol, atol=self.atol
            ):
                return value
            previous = value
        raise ArithmeticError("Intensity quadrature did not converge")

    def _values(self, times, marks, origin, delays):
        v = np.asarray(
            self.intensities(times, marks, origin, np.asarray(delays)), dtype=float
        )
        if (
            v.shape != (len(delays), self.num_event_types)
            or not np.isfinite(v).all()
            or np.any(v < 0)
        ):
            raise ValueError("Invalid marked intensities")
        return v

    def restricted_mean(self, times, marks, origin, cap):
        def survival(delays):
            return np.array(
                [
                    math.exp(
                        -self._integrate(
                            lambda u: self._values(times, marks, origin, u).sum(1),
                            float(d),
                        )
                    )
                    for d in delays
                ]
            )

        return self._integrate(survival, cap)

    def log_density(self, times, marks, horizon):
        origin, value = 0.0, 0.0
        times = torch.as_tensor(times).double()
        marks = torch.as_tensor(marks).long()
        for i in range(len(times) + 1):
            end = float(times[i]) if i < len(times) else horizon
            gap = end - origin
            if gap <= 0:
                raise ValueError(
                    "Positive intervals including terminal exposure required"
                )
            value -= self._integrate(
                lambda u: self._values(times[:i], marks[:i], origin, u).sum(1), gap
            )
            if i < len(times):
                intensity = self._values(times[:i], marks[:i], origin, np.array([gap]))[
                    0, int(marks[i])
                ]
                if intensity <= 0:
                    raise ValueError("Zero density at observed event")
                value += math.log(intensity)
            origin = end
        return value

    def papangelou(self, u, mark, times, marks, horizon):
        t = torch.as_tensor(times).double().cpu()
        m = torch.as_tensor(marks).long().cpu()
        if (
            not 0 < u < horizon
            or (t == u).any()
            or not 0 <= mark < self.num_event_types
        ):
            raise ValueError("Invalid event insertion")
        pos = int(torch.searchsorted(t, u))
        added_t = torch.cat([t[:pos], t.new_tensor([u]), t[pos:]])
        added_m = torch.cat([m[:pos], m.new_tensor([mark]), m[pos:]])
        ratio = math.exp(
            self.log_density(added_t, added_m, horizon)
            - self.log_density(t, m, horizon)
        )
        if not math.isfinite(ratio):
            raise ArithmeticError("Papangelou ratio overflow")
        return ratio


class PoissonReference(IntegratedIntensityReference):
    def __init__(self, rates, **kwargs):
        super().__init__(**kwargs)
        self.rates = np.atleast_1d(np.array(rates, dtype=float))
        if not np.isfinite(self.rates).all() or np.any(self.rates <= 0):
            raise ValueError("Strictly positive marked rates required")
        self.num_event_types = len(self.rates)

    def intensities(self, times, marks, origin, delays):
        return np.broadcast_to(self.rates, (len(delays), len(self.rates)))

    def restricted_mean(self, times, marks, origin, cap):
        return -np.expm1(-self.rates.sum() * cap) / self.rates.sum()

    def log_density(self, times, marks, horizon):
        return float(
            np.log(self.rates[np.asarray(marks, dtype=int)]).sum()
            - self.rates.sum() * horizon
        )

    def papangelou(self, u, mark, times, marks, horizon):
        return float(self.rates[mark])


class HawkesReference(IntegratedIntensityReference):
    """Unmarked empty-past Hawkes; rate is the stationary rate parameter."""

    num_event_types = 1

    def __init__(
        self,
        *,
        rate=100.0,
        branching=0.5,
        beta=10.0,
        exponent=2.0,
        kind="exp",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if (
            kind not in {"exp", "power"}
            or not np.isfinite([rate, branching, beta, exponent]).all()
            or rate <= 0
            or beta <= 0
            or not 0 <= branching < 1
            or exponent <= 1
        ):
            raise ValueError("Invalid Hawkes parameters")
        self.rate, self.branching, self.beta, self.exponent, self.kind = (
            rate,
            branching,
            beta,
            exponent,
            kind,
        )

    def intensities(self, times, marks, origin, delays):
        ages = origin - np.asarray(torch.as_tensor(times).cpu())
        z = np.asarray(delays)[:, None] + ages[None, :]
        if np.any(z < 0):
            raise ValueError("Future entered history")
        g = (
            self.branching
            * self.beta
            * (
                np.exp(-self.beta * z)
                if self.kind == "exp"
                else (self.exponent - 1) * (1 + self.beta * z) ** (-self.exponent)
            )
        )
        return (self.rate * (1 - self.branching) + g.sum(1))[:, None]

    def restricted_mean(self, times, marks, origin, cap):
        """Same survival integral, using the analytic integrated Hawkes kernel."""
        ages = origin - np.asarray(torch.as_tensor(times).cpu())
        if np.any(ages < 0):
            raise ValueError("Future entered history")

        def survival(delays):
            u = np.asarray(delays)[:, None]
            if self.kind == "exp":
                excitation = (
                    self.branching
                    * np.exp(-self.beta * ages)[None, :]
                    * (-np.expm1(-self.beta * u))
                )
            else:
                base = 1 + self.beta * ages
                excitation = (
                    self.branching
                    * base[None, :] ** (1 - self.exponent)
                    * (
                        -np.expm1(
                            (1 - self.exponent)
                            * np.log1p(self.beta * u / base[None, :])
                        )
                    )
                )
            return np.exp(
                -self.rate * (1 - self.branching) * np.asarray(delays)
                - excitation.sum(1)
            )

        return self._integrate(survival, cap)


class NeuralIntensityReference(IntegratedIntensityReference):
    """Public causal intensity adapter, including empty history and terminal survival.

    Model must already be loaded by its current class, frozen and in eval mode.
    Unsupported empty-history interfaces fail explicitly; no artificial BOS event.
    """

    def __init__(self, model, *, history_times=None, history_marks=None, **kwargs):
        super().__init__(**kwargs)
        if model.training:
            raise ValueError("Call model.eval() before constructing a frozen reference")
        self.model, self.num_event_types = model, model.num_event_types
        self.history_times = np.array(
            [] if history_times is None else history_times, dtype=float
        )
        self.history_marks = np.array(
            [] if history_marks is None else history_marks, dtype=int
        )
        if (
            self.history_times.ndim != 1
            or self.history_marks.shape != self.history_times.shape
            or not np.isfinite(self.history_times).all()
            or np.any(self.history_times < 0)
            or np.any(np.diff(self.history_times) <= 0)
        ):
            raise ValueError("Ordered nonnegative fixed conditioning history required")
        if np.any(
            (self.history_marks < 0) | (self.history_marks >= self.num_event_types)
        ):
            raise ValueError("Invalid conditioning marks")

    def intensities(self, times, marks, origin, delays):
        if self.model.training:
            raise ValueError("Reference changed to training mode")
        parameter = next(self.model.parameters())
        offset = float(self.history_times[-1]) if len(self.history_times) else 0.0
        t = torch.cat(
            [
                torch.as_tensor(
                    self.history_times, dtype=parameter.dtype, device=parameter.device
                ),
                torch.as_tensor(times, dtype=parameter.dtype, device=parameter.device)
                + offset,
            ]
        )[None, :]
        m = torch.cat(
            [
                torch.as_tensor(
                    self.history_marks, dtype=torch.long, device=parameter.device
                ),
                torch.as_tensor(marks, dtype=torch.long, device=parameter.device),
            ]
        )[None, :]
        d = torch.diff(t, dim=1, prepend=t.new_zeros(1, 1))
        length = max(1, t.shape[1])
        samples = torch.zeros(
            1, length, len(delays), dtype=parameter.dtype, device=parameter.device
        )
        elapsed = origin + offset - (float(t[0, -1]) if t.numel() else 0.0)
        samples[:, -1] = torch.as_tensor(
            delays + elapsed, dtype=parameter.dtype, device=parameter.device
        )
        try:
            with torch.no_grad():
                values = self.model.compute_intensities_at_sample_dtimes(
                    time_seqs=t,
                    time_delta_seqs=d,
                    type_seqs=m,
                    valid_event_mask=torch.ones_like(m, dtype=torch.bool),
                    sample_dtimes=samples,
                    compute_last_step_only=True,
                )
        except (RuntimeError, IndexError) as error:
            if not t.numel():
                raise ValueError(
                    "Model cannot score empty history; supply an explicit fixed conditioning history or mark this test unsupported"
                ) from error
            raise
        expected = (1, 1, len(delays), self.num_event_types)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Intensity model must return {expected}, got {tuple(values.shape)}"
            )
        return values[0, 0].double().cpu().numpy()
