from new_ltpp.configs import StatisticalTestConfig
import torch

from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.evaluation.statistical_testing.point_process_metric import MMD
from new_ltpp.evaluation.statistical_testing.point_process_kernels.kernel_protocol import (
    IPointProcessKernel,
)
from new_ltpp.models.model_protocol import ISimulableModel
from new_ltpp.shared_types import Batch
from new_ltpp.models.simulation.simulator import Simulator

from .base_test import ITest, FinalTestResult, TestStatistics


class MMDTwoSampleTest:
    """MMD two-sample permutation test for temporal point processes.

    Tests whether two sets of event sequences come from the same distribution.
    Uses a permutation test to estimate the p-value.

    The null hypothesis H0 is: the two distributions are the same.
    If p-value < threshold, we reject H0 (distributions are different).
    """

    kernel: IPointProcessKernel

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
        self.n_samples = n_samples
        self.mmd = MMD(kernel=kernel)

        self.total_observed_mmd: torch.Tensor | None = None
        self.total_perm_mmds: torch.Tensor | None = None  # (n_permutations,)
        self.total_count_ge: int = 0
        self.total_n_permutations: int = 0
        self.n_batches: int = 0

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

    def _accumulate(self, observed_mmd: torch.Tensor, perm_mmds: torch.Tensor) -> None:
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

    def get_final_p_value(self) -> torch.Tensor:
        if (
            self.n_batches == 0
            or self.total_perm_mmds is None
            or self.total_observed_mmd is None
        ):
            return torch.tensor(1.0, dtype=torch.float32)  # No data, p-value is 1

        perm_mmds = self.total_perm_mmds.to(self.total_observed_mmd.device)
        count_ge = (perm_mmds >= self.total_observed_mmd).sum()
        return (count_ge + 1) / (self.n_samples + 1)

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

        observed_mmd = self.mmd(batch_x, batch_y)
        pooled = self._concat_batches(batch_x, batch_y)

        perm_mmds: list[torch.Tensor] = []
        for _ in range(self.n_samples):
            perm = torch.randperm(total, device=observed_mmd.device)
            perm_x = self._select_batch(pooled, perm[:n])
            perm_y = self._select_batch(pooled, perm[n:])
            perm_mmds.append(self.mmd(perm_x, perm_y))

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

    def compute_statistics(
        self,
        batch_x: Batch,
        batch_y: Batch,
        accumulate: bool = True,
    ) -> TestStatistics:
        observed_mmd, perm_mmds = self._permutation_test(batch_x, batch_y)
        count_ge = (perm_mmds >= observed_mmd).sum()
        p_value = (count_ge + 1.0) / (self.n_samples + 1.0)

        if accumulate:
            self._accumulate(observed_mmd, perm_mmds)

        return TestStatistics(
            p_value=p_value,
            observed_statistic=observed_mmd,
            permuted_statistics=perm_mmds,
        )

    def test_model(
        self,
        model: ISimulableModel,
        data_loader: TypedDataLoader,
        statistical_test_config: StatisticalTestConfig,
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

        all_p_values = []
        all_mmds = []
        all_perm_mmds = []
        simulator = Simulator(
            model=model,
            statistical_test_config=statistical_test_config,
        )

        for batch in data_loader:
            simulated = simulator.simulate(batch=batch)
            test_stats = self.compute_statistics(batch_x=batch, batch_y=simulated)

            all_p_values.append(test_stats["p_value"].item())
            all_mmds.append(test_stats["observed_statistic"].item())
            all_perm_mmds.extend(test_stats["permuted_statistics"].tolist())

        p_val = self.get_final_p_value().item()

        return FinalTestResult(
            p_value=p_val,
            all_p_values=all_p_values,
            all_statistics=all_mmds,
            all_permuted_statistics=all_perm_mmds,
        )

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
        all_p_values = []
        all_mmds = []
        all_perm_mmds = []

        for batch_x, batch_y in zip(simulation, ground_truth):
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
        embedding_type="linear",
        num_discretization_points=100,
        dyadic_order=3,
        num_event_types=10,
    )

    test: ITest = MMDTwoSampleTest(kernel=kernel, n_samples=10)
