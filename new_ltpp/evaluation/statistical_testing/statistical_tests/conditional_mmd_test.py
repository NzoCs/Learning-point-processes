"""Conditional MMD: exchange continuations only within their shared history.

H0: conditional on fixed histories and a model fitted independently, X_i and
Y_i have the same law and are independent within each pair. Independent pairs
may have different laws. Batches must contain disjoint pairs. Fixed observation
design and matched time origins are the caller's responsibility.
"""

import math

import torch

from new_ltpp.shared_types import Batch

from .mmd_test import MMDTwoSampleTest

CONDITIONAL_MMD_PROTOCOL = "within_history_pair_swaps_joint_kernel_v1"
BATCH_AGGREGATION_PROTOCOL = "sum_statistics_product_randomization_v1"


def split_history_future(
    batch: Batch,
    history_events: int,
    future_events: int,
    *,
    allow_stopped=False,
    pad_token_id=0,
):
    """Select fixed valid-event counts and put the held-out future on origin zero.

    No filtering by future duration, no mutation, no fictitious padding events.
    A missing event raises rather than silently selecting a biased subset.
    """
    if history_events < 1 or future_events < 1:
        raise ValueError("History and future event counts must be positive")
    total = history_events + future_events
    rows_t, rows_m, rows_valid = [], [], []
    for row in range(len(batch.time_seqs)):
        mask = batch.valid_event_mask[row]
        t, m = batch.time_seqs[row][mask], batch.type_seqs[row][mask]
        if len(t) < total:
            if (
                len(t) < history_events
                or not allow_stopped
                or batch.no_next_event_mask is None
                or not batch.no_next_event_mask[row]
            ):
                raise ValueError(
                    "Not enough valid events for the requested holdout; a shorter future requires explicit no_next_event=True"
                )
        if not torch.isfinite(t).all() or (t < 0).any() or (torch.diff(t) < 0).any():
            raise ValueError(
                "Valid event times must be finite, nonnegative and ordered"
            )
        length = min(len(t), total)
        rows_t.append(
            torch.cat(
                (t[:total], torch.zeros(total - length, device=t.device, dtype=t.dtype))
            )
        )
        rows_m.append(
            torch.cat(
                (
                    m[:total],
                    torch.full(
                        (total - length,), pad_token_id, device=m.device, dtype=m.dtype
                    ),
                )
            )
        )
        rows_valid.append(torch.arange(total, device=t.device) < length)
    times, marks = torch.stack(rows_t), torch.stack(rows_m)
    valid = torch.stack(rows_valid)
    prefix_t = times[:, :history_events]
    future_t = times[:, history_events:] - prefix_t[:, -1:]
    future_t = future_t.masked_fill(~valid[:, history_events:], 0)

    def build(t, m, mask=None, stopped=None):
        mask = torch.ones_like(t, dtype=torch.bool) if mask is None else mask
        gaps = torch.diff(t, dim=1, prepend=torch.zeros_like(t[:, :1])).masked_fill(
            ~mask, 0
        )
        return Batch(
            t,
            gaps,
            m,
            mask,
            no_next_event_mask=stopped,
            sequence_ids=(
                None if batch.sequence_ids is None else list(batch.sequence_ids)
            ),
        )

    return build(prefix_t, marks[:, :history_events]), build(
        future_t,
        marks[:, history_events:],
        valid[:, history_events:],
        valid[:, history_events:].sum(-1) < future_events if allow_stopped else None,
    )


def pair_swap_indices(n: int, device=None):
    """Uniform element of the product of n two-element swap groups."""
    rows = torch.arange(n, device=device)
    swaps = torch.randint(2, (n,), device=device)
    return rows + n * swaps, rows + n * (1 - swaps)


def paired_joint_statistic(gram, x, y):
    """Off-diagonal paired U-statistic; exclude same-pair terms in all four blocks."""
    n = len(x)
    if n < 2:
        raise ValueError("At least two independent pairs are required")
    delta = gram[x][:, x] + gram[y][:, y] - gram[x][:, y] - gram[y][:, x]
    return (delta.sum() - delta.diag().sum()) / (n * (n - 1))


class ConditionalMMDTwoSampleTest(MMDTwoSampleTest):
    null_method = CONDITIONAL_MMD_PROTOCOL

    def __init__(
        self, kernel, n_samples, *, stopped_event_budget=None, observation_horizon=None
    ):
        super().__init__(kernel, n_samples)
        if observation_horizon is not None:
            if not math.isfinite(observation_horizon) or observation_horizon <= 0:
                raise ValueError("Observation horizon must be finite and positive")
            if stopped_event_budget is not None:
                raise ValueError(
                    "Choose one observation design: time window or stopped event budget"
                )
        self.observation_horizon = observation_horizon
        if stopped_event_budget is not None and stopped_event_budget < 1:
            raise ValueError("Stopped continuation event budget must be positive")
        self.stopped_event_budget = stopped_event_budget
        self.observation_protocol = (
            "fixed_time_window_conditional_v1"
            if observation_horizon is not None
            else (
                "complete_stopped_event_budget_v1"
                if stopped_event_budget is not None
                else "pairwise_truncation_recomputed_v1"
            )
        )

    def _validate_window_continuation(self, batch):
        times = batch.time_seqs[batch.valid_event_mask]
        if (
            not torch.isfinite(times).all()
            or (times < 0).any()
            or (times > self.observation_horizon).any()
        ):
            raise ValueError(
                "Window continuations must contain only events inside the fixed horizon"
            )
        if batch.no_next_event_mask is not None and batch.no_next_event_mask.any():
            raise ValueError(
                "Finite-window observations must not claim permanent termination"
            )

    def _validate_stopped_continuation(self, batch):
        flag = batch.no_next_event_mask
        n = len(batch.time_seqs)
        if flag is None or flag.shape != (n,) or flag.dtype != torch.bool:
            raise ValueError(
                "Stopped continuations require one boolean no_next_event flag per row"
            )
        counts = batch.valid_event_mask.sum(-1)
        if (counts > self.stopped_event_budget).any() or not torch.equal(
            flag.to(counts.device), counts < self.stopped_event_budget
        ):
            raise ValueError(
                "Each continuation must reach the event budget or explicitly terminate"
            )

    def compute_statistics(
        self, batch_x, batch_y, accumulate=True, *, histories=None, pair_ids=None
    ):
        self.validate_pairs(batch_x, batch_y, histories)
        n = len(batch_x.time_seqs)
        if n < 2:
            raise ValueError(
                "At least two matched history/continuation pairs are required"
            )
        ids = self._check_pair_ids(pair_ids, n, accumulate)
        gram = self.joint_kernel_matrix(histories, batch_x, batch_y)
        rows = torch.arange(n, device=gram.device)
        observed = paired_joint_statistic(gram, rows, rows + n)
        null = torch.stack(
            [
                paired_joint_statistic(gram, *pair_swap_indices(n, gram.device))
                for _ in range(self.n_samples)
            ]
        )
        return self._finish_statistics(observed, null, accumulate, ids)

    def validate_pairs(self, batch_x, batch_y, histories):
        """Validate observations before buffering or computing their kernel."""
        if histories is None:
            raise ValueError("Conditional tests require the shared histories")
        n = len(batch_x.time_seqs)
        if n < 1 or n != len(batch_y.time_seqs) or n != len(histories.time_seqs):
            raise ValueError("Nonempty matched history/continuation pairs are required")
        for batch in (batch_x, batch_y):
            if self.stopped_event_budget is not None:
                self._validate_stopped_continuation(batch)
            if self.observation_horizon is not None:
                self._validate_window_continuation(batch)
            if batch.time_seqs.shape != batch.valid_event_mask.shape:
                raise ValueError(
                    "Continuation mask and event times must have the same shape"
                )
            for row in range(n):
                t = batch.time_seqs[row][batch.valid_event_mask[row]]
                if (
                    (
                        not len(t)
                        and self.observation_horizon is None
                        and (
                            batch.no_next_event_mask is None
                            or not batch.no_next_event_mask[row]
                        )
                    )
                    or not torch.isfinite(t).all()
                    or (t < 0).any()
                    or (torch.diff(t) < 0).any()
                ):
                    raise ValueError(
                        "Continuations must contain ordered finite relative times"
                    )
                gaps = batch.time_delta_seqs[row][batch.valid_event_mask[row]]
                expected = torch.diff(t, prepend=torch.zeros_like(t[:1]))
                if not torch.allclose(gaps, expected, rtol=1e-5, atol=1e-5):
                    raise ValueError(
                        "Continuation gaps must use the shared history endpoint as origin"
                    )

    def joint_kernel_matrix(self, histories, batch_x, batch_y):
        """Product kernel on history and future, prepared once on a fixed orbit.

        History similarity is a unit-bandwidth Gaussian of the RKHS distance
        from the supplied history kernel. This removes the signature kernel's
        common level-zero component without dropping the marginal information.
        No bandwidth is selected from labels or tuned against calibration results.
        """
        if self.observation_horizon is not None:
            self._validate_window_continuation(batch_x)
            self._validate_window_continuation(batch_y)
            px, py = batch_x, batch_y
        elif self.stopped_event_budget is None:
            px, py = self._truncate_to_min_time(batch_x, batch_y)
        else:
            self._validate_stopped_continuation(batch_x)
            self._validate_stopped_continuation(batch_y)
            px, py = batch_x, batch_y
        pool = self._concat_batches(px, py)
        future = self.kernel.compute_gram_matrix(pool, pool)
        if self.stopped_event_budget is not None:
            stopped = torch.cat(
                (batch_x.no_next_event_mask, batch_y.no_next_event_mask)
            ).to(future)
            # Product with a unit-bandwidth Gaussian on the actual terminal flag.
            # Complete futures and pooled normalization commute with pair swaps.
            future = future * torch.exp(
                -0.5 * (stopped[:, None] - stopped[None, :]).square()
            )
        n = len(histories.time_seqs)
        if histories.time_seqs.shape[1] == 0:
            weight = torch.ones((n, n), dtype=future.dtype, device=future.device)
        else:
            history = self.kernel.compute_gram_matrix(histories, histories)
            distance = (
                history.diag()[:, None] + history.diag()[None, :] - 2 * history
            ).clamp_min(0)
            weight = torch.exp(-0.5 * distance)
        # Each pair has the same history for its two continuations. Truncation,
        # pooled normalization and this product commute with within-pair swaps.
        return future * weight.repeat(2, 2)

    def test_simulation(self, simulation, ground_truth):
        raise ValueError(
            "Conditional loaders require explicit histories; use compute_statistics"
        )

    def test_model(
        self,
        model,
        data_loader,
        statistical_test_config,
        *,
        history_events=None,
        future_events=None,
    ):
        if history_events is None or future_events is None:
            raise ValueError(
                "Model calibration requires history_events and future_events"
            )
        if self.observation_horizon is not None:
            raise ValueError(
                "Time-window tests require explicitly window-observed futures via compute_statistics"
            )
        if (
            self.stopped_event_budget is not None
            and future_events != self.stopped_event_budget
        ):
            raise ValueError(
                "Future event quota must match the stopped observation budget"
            )
        from new_ltpp.models.simulation.simulator import Simulator

        if getattr(model, "training", False):
            raise ValueError(
                "Conditional calibration requires a frozen evaluation model"
            )
        self.reset_accumulators()
        if getattr(model, "_simulator", None) is None:
            model._simulator = Simulator(model)
        values, statistics, null = [], [], []
        for batch in data_loader:
            histories, future = split_history_future(
                batch,
                history_events,
                future_events,
                allow_stopped=self.stopped_event_budget is not None,
                pad_token_id=getattr(model, "pad_token_id", 0),
            )
            simulated = model.simulate(histories, num_events_to_simulate=future_events)
            result = self.compute_statistics(
                future, simulated, histories=histories, pair_ids=batch.sequence_ids
            )
            values.append(float(result["p_value"]))
            statistics.append(float(result["observed_statistic"]))
            null.extend(result["permuted_statistics"].tolist())
        return dict(
            p_value=float(self.get_final_p_value()),
            all_p_values=values,
            all_statistics=statistics,
            all_permuted_statistics=null,
        )
