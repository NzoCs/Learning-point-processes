"""Complete event-budget observations; quiet windows are not terminal events."""

import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels.kernel_protocol import (
    PointProcessKernel,
)
from new_ltpp.shared_types import (
    STOPPED_OBSERVATION_PROTOCOL,
    Batch,
    uses_stopped_event_budget,
)

from .mmd_test import MMDTwoSampleTest


def validate_stopped_budget(batch, budget):
    flag = batch.no_next_event_mask
    n = len(batch.time_seqs)
    if flag is None or flag.shape != (n,) or flag.dtype != torch.bool:
        raise ValueError(
            "Stopped continuations require one boolean no_next_event flag per row"
        )
    counts = batch.valid_event_mask.sum(-1)
    if (counts > budget).any() or not torch.equal(
        flag.to(counts.device), counts < budget
    ):
        raise ValueError(
            "Each continuation must reach the event budget or explicitly terminate"
        )


def observe_event_budget(batch, budget):
    """Known complete quota or explicitly known stop; never infer stop from padding."""
    counts = batch.valid_event_mask.sum(-1)
    short = counts < budget
    if short.any() and (
        batch.no_next_event_mask is None
        or (short & ~batch.no_next_event_mask.to(short.device)).any()
    ):
        raise ValueError(
            "A future shorter than the event budget requires explicit no_next_event=True; censoring is not termination"
        )
    # Keep the first valid events if a source contains more than the quota.
    mask = batch.valid_event_mask & (batch.valid_event_mask.long().cumsum(-1) <= budget)
    return Batch(
        batch.time_seqs,
        batch.time_delta_seqs,
        batch.type_seqs,
        mask,
        sequence_ids=batch.sequence_ids,
        no_next_event_mask=short,
    )


class StoppedKernel(PointProcessKernel):
    def __init__(self, kernel):
        self.kernel = kernel

    def compute_gram_matrix(self, x, y):
        gram = self.kernel.compute_gram_matrix(x, y)
        a, b = x.no_next_event_mask.to(gram), y.no_next_event_mask.to(gram)
        return gram * torch.exp(-0.5 * (a[:, None] - b[None, :]).square())

    def compute_mmd(self, x, y):
        # One pooled normalization, shared by all four blocks of this statistic.
        pool = MMDTwoSampleTest(self.kernel, 1)._concat_batches(x, y)
        gram = self.compute_gram_matrix(pool, pool)
        n, m = len(x.time_seqs), len(y.time_seqs)
        if min(n, m) < 2:
            raise ValueError("Unbiased MMD requires at least two samples in each batch")
        xx, yy = gram[:n, :n], gram[n:, n:]
        return (
            (xx.sum() - xx.trace()) / (n * (n - 1))
            + (yy.sum() - yy.trace()) / (m * (m - 1))
            - 2 * gram[:n, n:].mean()
        )


class StoppedMMDTwoSampleTest(MMDTwoSampleTest):
    """Unconditional complete comparison; validity still requires exchangeability."""

    observation_protocol = STOPPED_OBSERVATION_PROTOCOL

    def __init__(self, kernel, n_samples):
        super().__init__(StoppedKernel(kernel), n_samples)
        self.event_budget = None

    def compute_statistics(self, batch_x, batch_y, accumulate=True, *, pair_ids=None):
        self.event_budget = batch_x.time_seqs.shape[1]
        return super().compute_statistics(
            batch_x, batch_y, accumulate, pair_ids=pair_ids
        )

    def _truncate_to_min_time(self, batch_x, batch_y):
        for batch in (batch_x, batch_y):
            validate_stopped_budget(batch, self.event_budget)
        return batch_x, batch_y
