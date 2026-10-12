"""Conditional MMD: exchange continuations only within their shared history.

H0: conditional on fixed histories and a model fitted independently, X_i and
Y_i have the same law and are independent within each pair. Independent pairs
may have different laws. Batches must contain disjoint pairs. Fixed observation
design and matched time origins are the caller's responsibility.
"""

import torch

from new_ltpp.shared_types import Batch

from .mmd_test import MMDTwoSampleTest

CONDITIONAL_MMD_PROTOCOL = "within_history_pair_swaps_v1"
BATCH_AGGREGATION_PROTOCOL = "sum_statistics_product_randomization_v1"


def split_history_future(batch: Batch, history_events: int, future_events: int):
    """Select fixed valid-event counts and put the held-out future on origin zero.

    No filtering by future duration, no mutation, no fictitious padding events.
    A missing event raises rather than silently selecting a biased subset.
    """
    if history_events < 1 or future_events < 1:
        raise ValueError("History and future event counts must be positive")
    total = history_events + future_events
    rows_t, rows_m = [], []
    for row in range(len(batch.time_seqs)):
        mask = batch.valid_event_mask[row]
        t, m = batch.time_seqs[row][mask], batch.type_seqs[row][mask]
        if len(t) < total:
            raise ValueError("Not enough valid events for the requested holdout")
        if not torch.isfinite(t).all() or (t < 0).any() or (torch.diff(t) < 0).any():
            raise ValueError(
                "Valid event times must be finite, nonnegative and ordered"
            )
        rows_t.append(t[:total])
        rows_m.append(m[:total])
    times, marks = torch.stack(rows_t), torch.stack(rows_m)
    prefix_t = times[:, :history_events]
    future_t = times[:, history_events:] - prefix_t[:, -1:]

    def build(t, m):
        return Batch(
            t,
            torch.diff(t, dim=1, prepend=torch.zeros_like(t[:, :1])),
            m,
            torch.ones_like(t, dtype=torch.bool),
        )

    return build(prefix_t, marks[:, :history_events]), build(
        future_t, marks[:, history_events:]
    )


def pair_swap_indices(n: int, device=None):
    """Uniform element of the product of n two-element swap groups."""
    rows = torch.arange(n, device=device)
    swaps = torch.randint(2, (n,), device=device)
    return rows + n * swaps, rows + n * (1 - swaps)


class ConditionalMMDTwoSampleTest(MMDTwoSampleTest):
    null_method = CONDITIONAL_MMD_PROTOCOL

    def compute_statistics(
        self, batch_x, batch_y, accumulate=True, *, histories=None, pair_ids=None
    ):
        if histories is None:
            raise ValueError("Conditional tests require the shared histories")
        n = len(batch_x.time_seqs)
        if n < 2 or n != len(batch_y.time_seqs) or n != len(histories.time_seqs):
            raise ValueError(
                "At least two matched history/continuation pairs are required"
            )
        for batch in (batch_x, batch_y):
            if batch.time_seqs.shape != batch.valid_event_mask.shape:
                raise ValueError(
                    "Continuation mask and event times must have the same shape"
                )
            for row in range(n):
                t = batch.time_seqs[row][batch.valid_event_mask[row]]
                if (
                    not len(t)
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
        return super().compute_statistics(
            batch_x, batch_y, accumulate, pair_ids=pair_ids
        )

    def _permutation_test(self, batch_x, batch_y):
        n = len(batch_x.time_seqs)
        pooled = self._concat_batches(batch_x, batch_y)
        observed = self.mmd(*self._truncate_to_min_time(batch_x, batch_y))
        null = []
        for _ in range(self.n_samples):
            x, y = pair_swap_indices(n, observed.device)
            # The pairing is fixed; min(T_xi,T_yi) commutes with within-pair swaps.
            px, py = self._select_batch(pooled, x), self._select_batch(pooled, y)
            null.append(self.mmd(*self._truncate_to_min_time(px, py)))
        return observed, torch.stack(null)

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
                batch, history_events, future_events
            )
            simulated = model.simulate(histories, num_events_to_simulate=future_events)
            result = self.compute_statistics(future, simulated, histories=histories)
            values.append(float(result["p_value"]))
            statistics.append(float(result["observed_statistic"]))
            null.extend(result["permuted_statistics"].tolist())
        return dict(
            p_value=float(self.get_final_p_value()),
            all_p_values=values,
            all_statistics=statistics,
            all_permuted_statistics=null,
        )
