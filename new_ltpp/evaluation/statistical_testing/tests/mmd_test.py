import torch

from new_ltpp.models.base_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.evaluation.statistical_testing.statistical_metrics import MMD
from new_ltpp.evaluation.statistical_testing.kernels.kernel_protocol import (
    PointProcessKernel,
)
from new_ltpp.shared_types import Batch

from .test_protocol import TestProtocol


class MMDTwoSampleTest(TestProtocol):
    """MMD two-sample permutation test for temporal point processes.

    Tests whether two sets of event sequences come from the same distribution.
    Uses a permutation test to estimate the p-value.

    The null hypothesis H0 is: the two distributions are the same.
    If p-value < threshold, we reject H0 (distributions are different).
    """

    kernel: PointProcessKernel

    def __init__(
        self,
        kernel: PointProcessKernel,
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

    @property
    def name(self) -> str:
        """Return the name of the test."""
        return self.__class__.__name__

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
    ) -> tuple[float, list[float]]:
        """Compute observed MMD² and per-permutation MMD² values for a batch pair.

        1. Compute the observed MMD² between batch_x and batch_y.
        2. Pool all sequences, then for each permutation:
           - Randomly split the pool into two groups of size n and m.
           - Compute MMD² for this random split.

        Args:
            batch_x: Batch of sequences from distribution X.
            batch_y: Batch of sequences from distribution Y.

        Returns:
            Tuple of (observed_mmd, perm_mmds) where perm_mmds is a list
            of MMD² values for each permutation.
        """
        n = batch_x.time_seqs.shape[0]
        m = batch_y.time_seqs.shape[0]
        total = n + m

        observed_mmd = self.mmd(batch_x, batch_y)
        pooled = self._concat_batches(batch_x, batch_y)

        perm_mmds: list[float] = []
        for _ in range(self.n_permutations):
            perm = torch.randperm(total)
            perm_x = self._select_batch(pooled, perm[:n])
            perm_y = self._select_batch(pooled, perm[n:])
            perm_mmds.append(self.mmd(perm_x, perm_y))

        return observed_mmd, perm_mmds

    def p_value_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> float:
        """Compute the p-value of the MMD two-sample permutation test
        on a single pair of batches.

        Useful for logging p-values during training.

        Args:
            batch_x: Batch of sequences from distribution X.
            batch_y: Batch of sequences from distribution Y.

        Returns:
            p-value of the permutation test.
        """
        observed_mmd, perm_mmds = self._permutation_test_from_batches(batch_x, batch_y)
        count_ge = sum(1 for pm in perm_mmds if pm >= observed_mmd)
        return (count_ge + 1) / (self.n_permutations + 1)

    def p_value_from_model(
        self,
        model: NeuralModel,
        data_loader: TypedDataLoader,
    ) -> float:
        """Compute the p-value of the MMD two-sample permutation test.

        Simulates sequences from the model for each batch and compares
        against the real sequences using a permutation test.

        Aggregates MMD² statistics across batches:
            T_obs = Σ_i MMD²_i(real_i, sim_i)
        For each permutation π:
            T*_π  = Σ_i MMD²_i(perm_x_i, perm_y_i)
        p-value = (#{T*_π >= T_obs} + 1) / (n_permutations + 1)

        Args:
            model: The neural point process model to evaluate.
            data_loader: DataLoader providing batches of real sequences.

        Returns:
            p-value of the aggregated permutation test.
        """
        # Process each batch immediately to avoid memory blow-up.
        # Store only scalar MMD values: observed + permuted per batch.
        t_obs = 0.0
        all_perm_mmds: list[list[float]] = []

        n_batches = 0
        for batch in data_loader:
            simulated = model.simulate(batch=batch)
            observed_mmd_i, perm_mmds_i = self._permutation_test_from_batches(
                batch, simulated
            )
            t_obs += observed_mmd_i
            all_perm_mmds.append(perm_mmds_i)
            n_batches += 1

        if n_batches == 0:
            return 1.0

        # Aggregate: T*_k = Σ_i perm_mmds[i][k], count how many >= T_obs
        count_ge = 0
        for k in range(self.n_permutations):
            t_perm = sum(all_perm_mmds[i][k] for i in range(n_batches))
            if t_perm >= t_obs:
                count_ge += 1

        return (count_ge + 1) / (self.n_permutations + 1)

    def p_value_from_dataloaders(
        self,
        data_loader_x: TypedDataLoader,
        data_loader_y: TypedDataLoader,
    ) -> float:
        """Compute the p-value of the MMD two-sample permutation test
        between two dataloaders (e.g. real vs pre-simulated).

        Aggregates MMD² statistics across batch pairs:
            T_obs = Σ_i MMD²_i(x_i, y_i)
        For each permutation π:
            T*_π  = Σ_i MMD²_i(perm_x_i, perm_y_i)
        p-value = (#{T*_π >= T_obs} + 1) / (n_permutations + 1)

        Args:
            data_loader_x: DataLoader providing batches of sequences from distribution X.
            data_loader_y: DataLoader providing batches of sequences from distribution Y.

        Returns:
            p-value of the aggregated permutation test.
        """
        # Process each batch pair immediately to avoid memory blow-up.
        t_obs = 0.0
        all_perm_mmds: list[list[float]] = []

        n_batches = 0
        for batch_x, batch_y in zip(data_loader_x, data_loader_y):
            observed_mmd_i, perm_mmds_i = self._permutation_test_from_batches(
                batch_x, batch_y
            )
            t_obs += observed_mmd_i
            all_perm_mmds.append(perm_mmds_i)
            n_batches += 1

        if n_batches == 0:
            return 1.0

        # Aggregate: T*_k = Σ_i perm_mmds[i][k], count how many >= T_obs
        count_ge = 0
        for k in range(self.n_permutations):
            t_perm = sum(all_perm_mmds[i][k] for i in range(n_batches))
            if t_perm >= t_obs:
                count_ge += 1

        return (count_ge + 1) / (self.n_permutations + 1)

    def statistic_from_batches(
        self,
        batch_x: Batch,
        batch_y: Batch,
    ) -> float:
        """Compute the MMD² statistic comparing two batches of samples.

        Args:
            batch_x: Batch of sequences from distribution X.
            batch_y: Batch of sequences from distribution Y.

        Returns:
            MMD² statistic value.
        """
        return self.mmd(batch_x, batch_y)
