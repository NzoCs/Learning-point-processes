import torch

from new_ltpp.models.base_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.evaluation.statistical_testing.statistical_metrics import MMD
from new_ltpp.evaluation.statistical_testing.kernels.kernel_protocol import (
    IPointProcessKernel,
)
from new_ltpp.shared_types import Batch

from .base_test import Test


class MMDTwoSampleTest(Test):
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
        n_permutations: int,
    ):
        """Initialize the MMD two-sample permutation test.

        Args:
            kernel: Kernel to use for MMD computation.
            n_permutations: Number of permutations for the permutation test.
        """
        self.kernel = kernel
        self.n_permutations = n_permutations
        self.mmd = MMD(kernel=kernel)

        self.total_observed_mmd: torch.Tensor = torch.tensor(0.0)
        self.total_perm_mmds: torch.Tensor | None = None  # (n_permutations,)
        self.n_batches: int = 0

    @property
    def name(self) -> str:
        """Return the name of the test."""
        return self.__class__.__name__

    def reset_accumulators(self) -> None:
        self.total_observed_mmd: torch.Tensor = torch.tensor(0.0)
        self.total_perm_mmds: torch.Tensor | None = None  # (n_permutations,)
        self.n_batches: int = 0

    def _accumulate(self, observed_mmd: torch.Tensor, perm_mmds: torch.Tensor) -> None:
        self.total_observed_mmd += observed_mmd
        if self.total_perm_mmds is None:
            self.total_perm_mmds = perm_mmds.detach().clone()
        else:
            self.total_perm_mmds += perm_mmds.detach()
        self.n_batches += 1

    def get_final_p_value(self) -> float:
        if self.n_batches == 0 or self.total_perm_mmds is None:
            return 1.0
        count_ge = (self.total_perm_mmds >= self.total_observed_mmd).sum()
        return (count_ge.item() + 1) / (self.n_permutations + 1)

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

    def _permutation_test_from_batches(
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
        for _ in range(self.n_permutations):
            perm = torch.randperm(total, device=observed_mmd.device)
            perm_x = self._select_batch(pooled, perm[:n])
            perm_y = self._select_batch(pooled, perm[n:])
            perm_mmds.append(self.mmd(perm_x, perm_y))

        return observed_mmd, torch.stack(perm_mmds)

    def statistics_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
        accumulate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the p-value of the MMD two-sample permutation test
        on a single pair of batches.

        Useful for logging p-values during training.

        Args:
            batch_x: Batch of sequences from distribution X.
            batch_y: Batch of sequences from distribution Y.
            accumulate: If True, accumulate MMD² and p-value for later combination.

        Returns:
            p-value of the permutation test as a Tensor.
        """
        observed_mmd, perm_mmds = self._permutation_test_from_batches(batch_x, batch_y)
        count_ge = (perm_mmds >= observed_mmd).sum()
        p_value = (count_ge + 1.0) / (self.n_permutations + 1.0)

        if accumulate:
            self.total_count_ge += int(count_ge.item())
            self.total_n_permutations += self.n_permutations

        return p_value, observed_mmd, perm_mmds

    def _aggregate_permutations(
        self,
        t_obs: torch.Tensor,
        all_perm_mmds: list[torch.Tensor],
        n_batches: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate per-batch permutation MMDs and calculate final p-value, obs, and perms."""
        if n_batches == 0:
            return (
                torch.tensor(1.0, device=t_obs.device),
                t_obs,
                torch.zeros(self.n_permutations, device=t_obs.device),
            )

        # Aggregate: T*_k = Σ_i perm_mmds[i][k], count how many >= T_obs
        stacked_perms = torch.stack(all_perm_mmds)  # (n_batches, n_permutations)
        t_perms = stacked_perms.sum(dim=0)  # (n_permutations,)

        count_ge = (t_perms >= t_obs).sum()
        p_value = (count_ge + 1.0) / (self.n_permutations + 1.0)

        return p_value, t_obs, t_perms

    def mmd_from_batches(
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

    def statistics_from_model(
        self,
        model: NeuralModel,
        data_loader: TypedDataLoader,
        accumulate: bool = True,
    ) -> tuple[float, list[float], list[float], list[float]]:
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

        for batch in data_loader:
            simulated = model.simulate(batch=batch)
            p_value_i, observed_mmd_i, perm_mmds_i = self.statistics_from_batches(
                batch, simulated
            )

            all_p_values.append(p_value_i.item())
            all_mmds.append(observed_mmd_i.item())
            all_perm_mmds.extend(perm_mmds_i.tolist())

        p_val = self.get_final_p_value()

        return p_val, all_p_values, all_mmds, all_perm_mmds

    def statistics_from_dataloaders(
        self,
        data_loader_x: TypedDataLoader,
        data_loader_y: TypedDataLoader,
        accumulate: bool = True,
    ) -> tuple[float, list[float], list[float], list[float]]:
        """Compute the p-value of the MMD two-sample permutation test for two data loaders, e.g. ground truth and simulation.

        Args:
            data_loader_x: Data loader for ground truth batches.
            data_loader_y: Data loader for simulated batches.
            accumulate: Whether to accumulate statistics.

        Returns:
            p-value as a float.
        """
        all_p_values = []
        all_mmds = []
        all_perm_mmds = []

        n_batches = 0
        for batch_x, batch_y in zip(data_loader_x, data_loader_y):
            p_value_i, observed_mmd_i, perm_mmds_i = self.statistics_from_batches(
                batch_x, batch_y
            )

            all_p_values.append(p_value_i.item())
            all_mmds.append(observed_mmd_i.item())
            all_perm_mmds.extend(perm_mmds_i.tolist())

            n_batches += 1

        return self.get_final_p_value(), all_p_values, all_mmds, all_perm_mmds
