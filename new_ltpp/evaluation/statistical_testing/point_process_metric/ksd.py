"""Stein-Papangelou Gram (Yang/Rao/Neville2019 eq7,12) for marked configurations.

The histogram-RBF kernel is a deliberate finite representation, not claimed
characteristic. Birth integration sums over every mark. Numerical failures raise.
"""

import math

import numpy as np
import torch

from ..observations import configurations


class KSD:
    def __init__(
        self,
        reference,
        *,
        horizon=1.0,
        bins=20,
        scale=10.0,
        sigma=1.0,
        quadrature_order=8,
        quadrature_rtol=2e-5,
        max_states=50000,
    ):
        if (
            bins < 1
            or quadrature_order < 2
            or max_states < 1
            or not 0 < quadrature_rtol < 1
        ):
            raise ValueError("Invalid Stein quadrature or capacity")
        if not all(math.isfinite(v) and v > 0 for v in [horizon, scale, sigma]):
            raise ValueError("Positive finite scales and horizon required")
        self.reference, self.horizon, self.bins = reference, horizon, bins
        self.scale, self.sigma = scale, sigma
        self.order, self.rtol, self.max_states = (
            quadrature_order,
            quadrature_rtol,
            max_states,
        )

    def _births(self, times, marks, order):
        num_marks = self.reference.num_event_types
        cuts = np.unique(
            np.r_[np.linspace(0, self.horizon, self.bins + 1), times.cpu().numpy()]
        )
        nodes, weights = np.polynomial.legendre.leggauss(order)
        output = np.zeros(self.bins * num_marks)
        for left, right in zip(cuts[:-1], cuts[1:]):
            for z, w in zip(nodes, weights):
                u = float((left + right) / 2 + (right - left) * z / 2)
                bin_id = min(self.bins - 1, int(u / self.horizon * self.bins))
                for mark in range(num_marks):
                    value = float(
                        self.reference.papangelou(u, mark, times, marks, self.horizon)
                    )
                    if not math.isfinite(value) or value < 0:
                        raise ArithmeticError("Invalid Papangelou intensity")
                    output[bin_id * num_marks + mark] += w * (right - left) / 2 * value
        return output

    def features(self, batch):
        rows = configurations(batch, self.horizon, self.reference.num_event_types)
        dimension = self.bins * self.reference.num_event_types
        if len(rows) * (2 * dimension + 1) > self.max_states:
            raise ValueError(
                "Stein state budget exceeded; choose explicit smaller bins/batch in the protocol"
            )
        counts, births = [], []
        for times, marks in rows:
            index = (
                torch.floor(times.cpu() / self.horizon * self.bins).long()
                * self.reference.num_event_types
                + marks.cpu()
            )
            count = torch.bincount(index, minlength=dimension).double().numpy()
            previous = self._births(times, marks, self.order)
            accepted = None
            for order in [self.order * 2, self.order * 4]:
                current = self._births(times, marks, order)
                if np.allclose(current, previous, rtol=self.rtol, atol=1e-8):
                    accepted = current
                    break
                previous = current
            if accepted is None:
                raise ArithmeticError("Stein birth quadrature did not converge")
            counts.append(count)
            births.append(accepted)
        count, birth = np.array(counts), np.array(births)
        shifts = np.r_[np.zeros((1, dimension)), np.eye(dimension), -np.eye(dimension)]
        states = (count[:, None, :] + shifts[None, :, :]) / self.scale
        weights = np.c_[-(birth.sum(1) + count.sum(1)), birth, count]
        return torch.as_tensor(states, device=batch.time_seqs.device), torch.as_tensor(
            weights, device=batch.time_seqs.device
        )

    def gram(self, batch):
        states, weights = self.features(batch)
        n, atoms, dimension = states.shape
        flat = states.reshape(-1, dimension)
        norm = flat.square().sum(1)
        gram = torch.empty(n, n, dtype=torch.float64, device=flat.device)
        with torch.no_grad():
            for i in range(n):
                distance = (
                    states[i].square().sum(1)[:, None]
                    + norm[None, :]
                    - 2 * states[i] @ flat.T
                ).clamp_min(0)
                kernel = torch.exp(-distance / (2 * self.sigma**2))
                gram[i] = ((weights[i] @ kernel).reshape(n, atoms) * weights).sum(1)
        if not torch.isfinite(gram).all():
            raise ArithmeticError("Nonfinite Stein Gram")
        if not torch.allclose(gram, gram.T, rtol=1e-7, atol=1e-7):
            raise ArithmeticError("Stein Gram is not symmetric")
        return (gram + gram.T) / 2
