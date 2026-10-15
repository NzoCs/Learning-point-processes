from itertools import zip_longest

import torch

from new_ltpp.configs import StatisticalTestConfig
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.evaluation.statistical_testing.point_process_kernels.kernel_protocol import (
    IPointProcessKernel,
)
from new_ltpp.evaluation.statistical_testing.point_process_metric import MMD
from new_ltpp.models.model_protocol import ISimulableModel
from new_ltpp.shared_types import Batch

from .base_test import FinalTestResult, ITest, TestStatistics

MMD_OBSERVATION_PROTOCOL = "pairwise_truncation_recomputed_v1"
MMD_P_VALUE_PROTOCOL = "plus_one_float64_conservative_ties_v1"


def monte_carlo_p_value(observed: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
    """Plus-one p-value, preserving the alpha boundary and numerical ties."""
    if observed.numel() != 1 or null.ndim != 1 or not null.numel():
        raise ValueError(
            "One observed statistic and a nonempty null vector are required"
        )
    if not torch.isfinite(observed).all() or not torch.isfinite(null).all():
        raise ValueError("Cannot compute p-values from nonfinite statistics")
    count = (
        (null >= observed) | torch.isclose(null, observed, rtol=1e-10, atol=1e-12)
    ).sum()
    return (count.to(dtype=torch.float64) + 1) / (null.numel() + 1)


class MMDTwoSampleTest:
    """MMD two-sample permutation test for temporal point processes.

    Tests whether two sets of event sequences come from the same distribution.
    Uses a permutation test to estimate the p-value.

    The null hypothesis H0 is: the two distributions are the same.
    If p-value < threshold, we reject H0 (distributions are different).
    """

    kernel: IPointProcessKernel
    null_method = "permutation"

    def __init__(
        self,
        kernel: IPointProcessKernel,
        n_samples: int,
    ):
        """Initialize the MMD two-sample permutation test.

        Args:
            kernel: Kernel to use for MMD computation.
            n_samples: Number of samples for the permutation test.
        """
        self.kernel = kernel
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        self.n_samples = n_samples
        self.mmd = MMD(kernel=kernel)

        self.total_observed_mmd: torch.Tensor | None = None
        self.total_perm_mmds: torch.Tensor | None = None  # (n_permutations,)
        self.total_count_ge: int = 0
        self.total_n_permutations: int = 0
        self.n_batches: int = 0
        self._seen_pair_ids: set = set()
        self._pair_id_mode: bool | None = None
        self._aggregation_null_method: str | None = None

    @property
    def name(self) -> str:
        """Return the name of the test."""
        return self.__class__.__name__

    def reset_accumulators(self) -> None:
        self.total_observed_mmd: torch.Tensor | None = None
        self.total_perm_mmds: torch.Tensor | None = None  # (n_permutations,)
        self.total_count_ge: int = 0
        self.total_n_permutations: int = 0
        self.n_batches: int = 0
        self._seen_pair_ids.clear()
        self._pair_id_mode = None
        self._aggregation_null_method = None

    def _check_pair_ids(self, pair_ids, n, accumulate):
        if pair_ids is None:
            if accumulate and self._pair_id_mode:
                raise ValueError("All accumulated batches must supply pair_ids")
            return None
        ids = set(pair_ids)
        if len(pair_ids) != n or len(ids) != n:
            raise ValueError("pair_ids must identify distinct independent pairs")
        if accumulate:
            if self._pair_id_mode is False:
                raise ValueError("All accumulated batches must supply pair_ids")
            if ids & self._seen_pair_ids:
                raise ValueError(
                    "Overlapping pairs cannot use independent batch randomizations"
                )
        return ids

    def _accumulate(self, observed_mmd: torch.Tensor, perm_mmds: torch.Tensor) -> None:
        if self.n_batches and self._aggregation_null_method != self.null_method:
            raise ValueError("Cannot mix null schemes; reset accumulators first")
        if observed_mmd.numel() != 1 or perm_mmds.ndim != 1 or not perm_mmds.numel():
            raise ValueError(
                "A scalar statistic and a nonempty null vector are required"
            )
        if (
            not torch.isfinite(observed_mmd).all()
            or not torch.isfinite(perm_mmds).all()
        ):
            raise ValueError("Cannot accumulate nonfinite statistics")
        if (
            self.total_perm_mmds is not None
            and self.total_perm_mmds.shape != perm_mmds.shape
        ):
            raise ValueError("Cannot pool batches with different null sample counts")
        if self.total_observed_mmd is None:
            self.total_observed_mmd = observed_mmd.detach().clone()
        else:
            self.total_observed_mmd += observed_mmd.detach().to(
                self.total_observed_mmd.device
            )

        if self.total_perm_mmds is None:
            self.total_perm_mmds = perm_mmds.detach().clone()
        else:
            self.total_perm_mmds += perm_mmds.detach().to(self.total_perm_mmds.device)
        self.n_batches += 1
        self._aggregation_null_method = self.null_method

    def get_final_p_value(self) -> torch.Tensor:
        if (
            self.n_batches == 0
            or self.total_perm_mmds is None
            or self.total_observed_mmd is None
        ):
            return torch.tensor(1.0, dtype=torch.float64)  # No data, p-value is 1

        perm_mmds = self.total_perm_mmds.to(self.total_observed_mmd.device)
        return monte_carlo_p_value(self.total_observed_mmd, perm_mmds)

    def _concat_batches(self, batch_x: Batch, batch_y: Batch) -> Batch:
        """Concatenate two batches along the batch dimension.

        Args:
            batch_x: First batch.
            batch_y: Second batch.

        Returns:
            Concatenated batch.
        """
        # Pad to same seq_len if needed
        L_x = batch_x.time_seqs.shape[1]
        L_y = batch_y.time_seqs.shape[1]
        L = max(L_x, L_y)

        def _pad(
            tensor: torch.Tensor, target_len: int, pad_value: float = 0.0
        ) -> torch.Tensor:
            if tensor.shape[1] == target_len:
                return tensor
            pad_size = target_len - tensor.shape[1]
            padding = torch.full(
                (tensor.shape[0], pad_size, *tensor.shape[2:]),
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return torch.cat([tensor, padding], dim=1)

        return Batch(
            time_seqs=torch.cat(
                [_pad(batch_x.time_seqs, L), _pad(batch_y.time_seqs, L)], dim=0
            ),
            time_delta_seqs=torch.cat(
                [_pad(batch_x.time_delta_seqs, L), _pad(batch_y.time_delta_seqs, L)],
                dim=0,
            ),
            type_seqs=torch.cat(
                [_pad(batch_x.type_seqs, L), _pad(batch_y.type_seqs, L)], dim=0
            ),
            valid_event_mask=torch.cat(
                [_pad(batch_x.valid_event_mask, L), _pad(batch_y.valid_event_mask, L)],
                dim=0,
            ),
            no_next_event_mask=(
                torch.cat((batch_x.no_next_event_mask, batch_y.no_next_event_mask))
                if batch_x.no_next_event_mask is not None
                and batch_y.no_next_event_mask is not None
                else None
            ),
        )

    def _select_batch(self, pooled: Batch, indices: torch.Tensor) -> Batch:
        """Select a subset of the pooled batch by indices.

        Args:
            pooled: Pooled batch of shape (n+m, L).
            indices: 1D tensor of indices to select.

        Returns:
            Selected sub-batch.
        """
        return Batch(
            time_seqs=pooled.time_seqs[indices],
            time_delta_seqs=pooled.time_delta_seqs[indices],
            type_seqs=pooled.type_seqs[indices],
            valid_event_mask=pooled.valid_event_mask[indices],
            no_next_event_mask=(
                None
                if pooled.no_next_event_mask is None
                else pooled.no_next_event_mask[indices]
            ),
        )

    def _permutation_test(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute observed MMD² and per-permutation MMD² values for a batch pair.

        1. Compute the observed MMD² between batch_x and batch_y.
        2. Pool all sequences, then for each permutation:
           - Randomly split the pool into two groups of size n and m.
           - Compute MMD² for this random split.

        Args:
            batch_x: Batch of sequences from distribution X.
            batch_y: Batch of sequences from distribution Y.

        Returns:
            Tuple of (observed_mmd, perm_mmds) as Tensors.
        """
        n = batch_x.time_seqs.shape[0]
        m = batch_y.time_seqs.shape[0]
        total = n + m

        observed_mmd = self.mmd(*self._truncate_to_min_time(batch_x, batch_y))
        # Pool RAW sequences: masks must be recomputed after every permutation.
        pooled = self._concat_batches(batch_x, batch_y)

        perm_mmds: list[torch.Tensor] = []
        for _ in range(self.n_samples):
            perm = torch.randperm(total, device=observed_mmd.device)
            perm_x = self._select_batch(pooled, perm[:n])
            perm_y = self._select_batch(pooled, perm[n:])
            perm_mmds.append(self.mmd(*self._truncate_to_min_time(perm_x, perm_y)))

        return observed_mmd, torch.stack(perm_mmds, dim=-1)

    def _mmd(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> torch.Tensor:
        """Compute the MMD² statistic comparing two batches of samples.

        Args:
            batch_x: Batch of sequences from distribution X.
            batch_y: Batch of sequences from distribution Y.

        Returns:
            MMD² statistic value as a Tensor.
        """
        return self.mmd(batch_x, batch_y)

    def _truncate_to_min_time(
        self, batch_x: Batch, batch_y: Batch
    ) -> tuple[Batch, Batch]:
        """Truncate sequences to the minimum time horizon between paired sequences.

        This ensures that when comparing sequences that have generated the same
        number of events, they are evaluated over the exact same time window.
        """
        if len(batch_x.time_seqs) != len(batch_y.time_seqs):
            raise ValueError("Pairwise truncation requires equal group sizes")
        # Calculate max time for each sequence in both batches
        T_max_x = (
            batch_x.time_seqs.masked_fill(~batch_x.valid_event_mask, 0.0)
            .max(dim=1)
            .values
        )
        T_max_y = (
            batch_y.time_seqs.masked_fill(~batch_y.valid_event_mask, 0.0)
            .max(dim=1)
            .values
        )

        # Element-wise minimum across the batch
        T_min = torch.min(T_max_x, T_max_y)

        # Update masks to only include events up to T_min
        mask_x = batch_x.valid_event_mask & (batch_x.time_seqs <= T_min.unsqueeze(1))
        mask_y = batch_y.valid_event_mask & (batch_y.time_seqs <= T_min.unsqueeze(1))

        trunc_x = Batch(
            time_seqs=batch_x.time_seqs,
            time_delta_seqs=batch_x.time_delta_seqs,
            type_seqs=batch_x.type_seqs,
            valid_event_mask=mask_x,
        )

        trunc_y = Batch(
            time_seqs=batch_y.time_seqs,
            time_delta_seqs=batch_y.time_delta_seqs,
            type_seqs=batch_y.type_seqs,
            valid_event_mask=mask_y,
        )

        return trunc_x, trunc_y

    def compute_statistics(
        self,
        batch_x: Batch,
        batch_y: Batch,
        accumulate: bool = True,
        *,
        pair_ids=None,
    ) -> TestStatistics:
        """Use the same pairwise observation rule for the observation and each null draw.

        Permutations start from raw sequences and recompute masks and kernel
        preparation. Calibration requires exchangeability under H0; conditional
        model simulations and pooled batches need separate scientific validation.
        """
        if (
            accumulate
            and self.n_batches
            and self._aggregation_null_method != self.null_method
        ):
            raise ValueError("Cannot mix null schemes; reset accumulators first")
        ids = self._check_pair_ids(pair_ids, len(batch_x.time_seqs), accumulate)
        observed_mmd, perm_mmds = self._permutation_test(batch_x, batch_y)

        return self._finish_statistics(observed_mmd, perm_mmds, accumulate, ids)

    def _finish_statistics(self, observed_mmd, perm_mmds, accumulate, ids):

        p_value = monte_carlo_p_value(observed_mmd, perm_mmds)

        if accumulate:
            self._accumulate(observed_mmd, perm_mmds)
            self._pair_id_mode = ids is not None
            if ids is not None:
                self._seen_pair_ids.update(ids)

        return TestStatistics(
            p_value=p_value,
            observed_statistic=observed_mmd,
            permuted_statistics=perm_mmds,
            num_null_samples=perm_mmds.numel(),
            null_method=self.null_method,
        )

    def test_model(
        self,
        model: ISimulableModel,
        data_loader: TypedDataLoader,
        statistical_test_config: StatisticalTestConfig,
        *,
        history_events: int | None = None,
        future_events: int | None = None,
    ) -> FinalTestResult:
        """Compute the p-value of the MMD two-sample permutation test for a trained model.

        Args:
            model: Model to simulate from.
            data_loader: Data loader for ground truth batches.
            accumulate: Whether to accumulate statistics.

        Returns:
            p-value: float
            all_p_values: list[float]
            all_mmds: list[float]
            all_perm_mmds: list[float]
        """

        # A history and its own continuation are not exchangeable observations.
        # Require an explicit held-out future and the conditional swap protocol.
        if history_events is None or future_events is None:
            raise ValueError(
                "Model calibration requires history_events and future_events"
            )
        from .conditional_mmd_test import ConditionalMMDTwoSampleTest

        self.reset_accumulators()
        conditional = ConditionalMMDTwoSampleTest(self.kernel, self.n_samples)
        result = conditional.test_model(
            model,
            data_loader,
            statistical_test_config,
            history_events=history_events,
            future_events=future_events,
        )
        self.total_observed_mmd = conditional.total_observed_mmd
        self.total_perm_mmds = conditional.total_perm_mmds
        self.n_batches = conditional.n_batches
        self._aggregation_null_method = conditional.null_method
        return result

    def test_simulation(
        self,
        simulation: TypedDataLoader,
        ground_truth: TypedDataLoader,
    ) -> FinalTestResult:
        """Compute the p-value of the MMD two-sample permutation test for two data loaders, e.g. ground truth and simulation.

        Args:
            simulation: Data loader for simulated batches.
            ground_truth: Data loader for ground truth batches.
            accumulate: Whether to accumulate statistics.

        Returns:
            p-value as a float.
        """
        self.reset_accumulators()
        all_p_values = []
        all_mmds = []
        all_perm_mmds = []

        for batch_x, batch_y in zip_longest(simulation, ground_truth):
            if batch_x is None or batch_y is None:
                self.reset_accumulators()
                raise ValueError("Data loaders must contain the same number of batches")
            test_stats = self.compute_statistics(batch_x=batch_x, batch_y=batch_y)

            all_p_values.append(test_stats["p_value"].item())
            all_mmds.append(test_stats["observed_statistic"].item())
            all_perm_mmds.extend(test_stats["permuted_statistics"].tolist())

        return FinalTestResult(
            p_value=self.get_final_p_value().item(),
            all_p_values=all_p_values,
            all_statistics=all_mmds,
            all_permuted_statistics=all_perm_mmds,
        )


if __name__ == "__main__":
    # Test the type compatibility of the test with the pipeline
    from new_ltpp.evaluation.statistical_testing.point_process_kernels import (
        SIGKernel,
    )
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
        LinearKernel,
    )

    kernel = SIGKernel(
        static_kernel=LinearKernel(),
        embedding_type="counting_grid",
        num_discretization_points=100,
        dyadic_order=3,
        num_event_types=10,
    )

    test: ITest = MMDTwoSampleTest(kernel=kernel, n_samples=10)
