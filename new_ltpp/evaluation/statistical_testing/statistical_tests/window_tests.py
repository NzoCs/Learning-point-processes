"""Full-Gram fixed-window MMD and paired relative prediction-loss tests.

No blockwise MMD averaging, pairwise horizon truncation, or event-level shuffles.
References and tuning must be independent of the evaluation labels.
"""

import math

import torch

from ..observations import configurations, from_configurations
from ..point_process_kernels.kernel_protocol import PointProcessKernel
from ..point_process_kernels.signature_backend import PySigLibKernel
from ..point_process_kernels.space_kernels import LinearKernel, RBFKernel
from .mmd_test import monte_carlo_p_value


def swap_signs(n, permutations, seed, *, device="cpu"):
    if n < 2 or permutations < 1:
        raise ValueError("At least two pairs and one randomization required")
    generator = torch.Generator().manual_seed(seed)
    return (1 - 2 * torch.randint(2, (permutations, n), generator=generator)).to(
        device=device, dtype=torch.float64
    )


def result(observed, null, method):
    return dict(
        p_value=monte_carlo_p_value(observed, null),
        observed_statistic=observed,
        permuted_statistics=null,
        num_null_samples=len(null),
        null_method=method,
    )


class FixedWindowSignatureKernel(PointProcessKernel):
    """Counting or exact-gap paths through the library's only signature backend.

    Marked gap vertices append one-hot event marks; terminal censor has zeros.
    Gap ranks retain event counts, and all gaps including terminal censor remain
    at original knots. Fixed scales must be selected before evaluation.
    """

    def __init__(
        self,
        *,
        horizon=1.0,
        num_event_types=1,
        representation="counting_grid",
        grid=100,
        count_scale=10.0,
        gap_scale=0.02,
        rank_scale=32.0,
        sigma=0.3,
        dyadic_order=0,
        max_batch=16,
        space_kernel_type="rbf",
    ):
        if representation not in {"counting_grid", "intervals", "events"}:
            raise ValueError("Unknown fixed-window representation")
        for v in [horizon, count_scale, gap_scale, rank_scale, sigma]:
            if not math.isfinite(v) or v <= 0:
                raise ValueError("Scales and horizon must be positive and finite")
        if grid < 2 or num_event_types < 1:
            raise ValueError("Invalid grid or mark count")
        self.horizon, self.num_event_types = horizon, num_event_types
        self.representation, self.grid = representation, grid
        self.count_scale, self.gap_scale, self.rank_scale = (
            count_scale,
            gap_scale,
            rank_scale,
        )
        if space_kernel_type not in {"linear", "rbf"}:
            raise ValueError("Unsupported static kernel")
        self.backend = PySigLibKernel(
            RBFKernel(sigma) if space_kernel_type == "rbf" else LinearKernel(),
            dyadic_order,
            max_batch,
        )

    def paths(self, batch):
        rows = configurations(batch, self.horizon, self.num_event_types)
        device = batch.time_seqs.device
        if self.representation == "counting_grid":
            grid = torch.linspace(
                0, self.horizon, self.grid, dtype=torch.float64, device=device
            )
            paths = []
            for times, marks in rows:
                total = (
                    torch.searchsorted(times.contiguous(), grid, right=True).double()
                    / self.count_scale
                )
                values = [total, grid]
                if self.num_event_types > 1:
                    values += [
                        torch.searchsorted(
                            times[marks == k].contiguous(), grid, right=True
                        ).double()
                        / self.count_scale
                        for k in range(self.num_event_types)
                    ]
                paths.append(torch.stack(values, dim=1))
            return torch.stack(paths)
        if self.representation == "events":
            width = 2 * max(len(t) for t, _ in rows) + 2
            channels = 2 + (self.num_event_types if self.num_event_types > 1 else 0)
            out = torch.zeros(
                len(rows), width, channels, dtype=torch.float64, device=device
            )
            for i, (times, marks) in enumerate(rows):
                point = torch.zeros(channels, dtype=torch.float64, device=device)
                for j, (t, mark) in enumerate(zip(times, marks)):
                    point = point.clone()
                    point[1] = t
                    out[i, 2 * j + 1] = point
                    point = point.clone()
                    point[0] += 1 / self.count_scale
                    if channels > 2:
                        point[2 + int(mark)] += 1 / self.count_scale
                    out[i, 2 * j + 2] = point
                point = point.clone()
                point[1] = self.horizon
                out[i, 2 * len(times) + 1 :] = point
            return out
        width = 3 * max(len(t) + 1 for t, _ in rows) + 1
        channels = 2 + (self.num_event_types if self.num_event_types > 1 else 0)
        out = torch.empty(
            len(rows), width, channels, dtype=torch.float64, device=device
        )
        for i, (times, marks) in enumerate(rows):
            gaps = torch.diff(
                torch.cat([times.new_zeros(1), times, times.new_tensor([self.horizon])])
            )
            knots = torch.zeros(
                len(gaps) + 1, channels, dtype=torch.float64, device=device
            )
            knots[:, 0] = torch.arange(len(knots), device=device) / self.rank_scale
            knots[1:, 1] = torch.log1p(gaps / self.gap_scale)
            if channels > 2 and len(times):
                knots[1:-1, 2:] = torch.nn.functional.one_hot(
                    marks, self.num_event_types
                ).double()
            j = torch.arange(3 * len(gaps), device=device)
            weight = (j % 3).double()[:, None] / 3
            out[i, : len(j)] = (1 - weight) * knots[j // 3] + weight * knots[j // 3 + 1]
            out[i, len(j) :] = knots[-1]
        return out

    def compute_gram_matrix(self, X, Y):
        a, b = self.paths(X), self.paths(Y)
        # A repeated endpoint has zero increment; keep one common padded length.
        width = max(a.shape[1], b.shape[1])

        def pad(p):
            return torch.cat([p, p[:, -1:].expand(-1, width - p.shape[1], -1)], dim=1)

        return self.backend.compute_Gram(pad(a), pad(b))


class CountRBFKernel(PointProcessKernel):
    """Count-only diagnostic, using the public RBF kernel on marked counts."""

    def __init__(self, *, horizon=1.0, num_event_types=1, count_scale=10.0, sigma=1.0):
        if num_event_types < 1 or not all(
            math.isfinite(v) and v > 0 for v in [horizon, count_scale, sigma]
        ):
            raise ValueError("Invalid count kernel parameters")
        self.horizon, self.num_event_types, self.count_scale = (
            horizon,
            num_event_types,
            count_scale,
        )
        self.kernel = RBFKernel(sigma)

    def compute_gram_matrix(self, X, Y):
        def counts(batch):
            rows = configurations(batch, self.horizon, self.num_event_types)
            return (
                torch.stack(
                    [
                        torch.bincount(m, minlength=self.num_event_types).double()
                        for _, m in rows
                    ]
                )[:, None, :]
                / self.count_scale
            )

        return self.kernel.Gram_matrix(counts(X), counts(Y))[:, :, 0, 0]


class FixedWindowMMDTest:
    """Paired joint-kernel MMD with fixed observation horizon, one pooled Gram."""

    name = "fixed_window_mmd"

    def __init__(self, kernel, *, n_samples=499, seed=42):
        if n_samples < 1:
            raise ValueError("Positive permutation count required")
        self.kernel, self.n_samples, self.seed = kernel, n_samples, seed

    def compute_statistics(self, batch_x, batch_y):
        rows_x = configurations(
            batch_x, self.kernel.horizon, self.kernel.num_event_types
        )
        rows_y = configurations(
            batch_y, self.kernel.horizon, self.kernel.num_event_types
        )
        n = len(rows_x)
        if n != len(rows_y) or n < 2:
            raise ValueError(
                "Equal groups with at least two independent pairs required"
            )
        pooled = from_configurations(rows_x + rows_y, device=batch_x.time_seqs.device)
        gram = self.kernel.compute_gram_matrix(pooled, pooled).double()
        delta = gram[:n, :n] + gram[n:, n:] - gram[:n, n:] - gram[n:, :n]
        delta = delta - torch.diag_embed(delta.diag())
        signs = swap_signs(n, self.n_samples, self.seed, device=gram.device)
        observed = delta.sum() / (n * (n - 1))
        null = ((signs @ delta) * signs).sum(1) / (n * (n - 1))
        return result(observed, null, "within_pair_trajectory_swaps")


class PredictiveLossTest:
    """Difference in mean all-origin relative MSE under a common frozen reference.

    This is the experimental RMSE baseline, not a universal law-equality test.
    No square root is taken. All trajectories have equal weight.
    """

    name = "relative_prediction_loss"

    def __init__(self, reference, *, horizon=1.0, n_samples=499, seed=42):
        if not math.isfinite(horizon) or horizon <= 0 or n_samples < 1:
            raise ValueError("Invalid horizon or permutation count")
        self.reference, self.horizon = reference, horizon
        self.n_samples, self.seed = n_samples, seed

    def losses(self, batch):
        values = []
        for times, marks in configurations(
            batch, self.horizon, self.reference.num_event_types
        ):
            errors = []
            origins = torch.cat([times.new_zeros(1), times])
            targets = torch.diff(torch.cat([origins, times.new_tensor([self.horizon])]))
            for i, t in enumerate(origins):
                cap = self.horizon - float(t)
                mu = float(
                    self.reference.restricted_mean(times[:i], marks[:i], float(t), cap)
                )
                if not math.isfinite(mu) or mu <= 0 or mu > cap * (1 + 1e-6):
                    raise ValueError(
                        "Invalid restricted mean; no clipped/excluded origins"
                    )
                errors.append(((float(targets[i]) - mu) / mu) ** 2)
            values.append(sum(errors) / len(errors))
        return torch.tensor(values, dtype=torch.float64)

    def compute_statistics(self, batch_x, batch_y):
        x, y = self.losses(batch_x), self.losses(batch_y)
        if len(x) != len(y) or len(x) < 2:
            raise ValueError(
                "At least two independent, exchangeable trajectory pairs required"
            )
        d = x - y
        signs = swap_signs(len(d), self.n_samples, self.seed)
        return result(
            d.mean().square(),
            (signs @ d / len(d)).square(),
            "within_pair_trajectory_swaps",
        )
