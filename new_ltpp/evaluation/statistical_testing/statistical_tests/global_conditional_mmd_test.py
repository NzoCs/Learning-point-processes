"""Global paired U-MMD with pooled paths and bounded Gram blocks.

Raw observations are retained on CPU, not N-by-N Grams. Every off-diagonal
pair is included, even across loader batches. Null signs follow sorted stable
sequence IDs and use a private generator, independent of simulation draws.
"""

import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.shared_types import Batch

from .conditional_mmd_test import ConditionalMMDTwoSampleTest
from .mmd_test import monte_carlo_p_value

GLOBAL_AGGREGATION_PROTOCOL = "global_paired_u_mmd_pooled_signature_blocks_v1"
GLOBAL_NULL_RNG_SALT = 0x474C4F42


def copy_rows(batch, indices=None):
    """Detached snapshot, including masks, terminal flags and ordered identities."""
    indices = slice(None) if indices is None else indices
    return Batch(
        *(
            getattr(batch, name)[indices].detach().cpu().clone()
            for name in (
                "time_seqs",
                "time_delta_seqs",
                "type_seqs",
                "valid_event_mask",
            )
        ),
        no_next_event_mask=(
            None
            if batch.no_next_event_mask is None
            else batch.no_next_event_mask[indices].detach().cpu().clone()
        ),
        sequence_ids=(
            None
            if batch.sequence_ids is None
            else (
                list(batch.sequence_ids)[indices]
                if isinstance(indices, slice)
                else [batch.sequence_ids[int(i)] for i in indices]
            )
        ),
    )


class GlobalConditionalMMDTwoSampleTest(ConditionalMMDTwoSampleTest):
    aggregation_method = GLOBAL_AGGREGATION_PROTOCOL

    def __init__(self, kernel, n_samples, *, block_size=64, **kwargs):
        if not isinstance(kernel, SIGKernel):
            raise ValueError("Global conditional collection requires sig_kernel")
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 1
        ):
            raise ValueError("Global Gram block size must be a positive integer")
        super().__init__(kernel, n_samples, **kwargs)
        self.block_size = block_size
        self._batches = []
        self._seed = None
        self._device = None

    def _check_pair_ids(self, pair_ids, n, accumulate):
        if pair_ids is None or any(not isinstance(i, str) for i in pair_ids):
            raise ValueError("Global conditional MMD requires stable string pair_ids")
        return super()._check_pair_ids(pair_ids, n, accumulate)

    def compute_statistics(
        self, batch_x, batch_y, accumulate=True, *, histories=None, pair_ids=None
    ):
        if not accumulate:
            return super().compute_statistics(
                batch_x, batch_y, False, histories=histories, pair_ids=pair_ids
            )
        self.validate_pairs(batch_x, batch_y, histories)
        n = len(batch_x.time_seqs)
        ids = self._check_pair_ids(pair_ids, n, True)
        for batch in (histories, batch_x, batch_y):
            if batch.sequence_ids is not None and list(batch.sequence_ids) != list(
                pair_ids
            ):
                raise ValueError(
                    "History and future sequence IDs must match pair_ids in order"
                )
            if batch.time_seqs.device != batch_x.time_seqs.device:
                raise ValueError("Matched observations must share a device")
        if self._device is not None and self._device != batch_x.time_seqs.device:
            raise ValueError("Cannot change devices during global collection")
        # These diagnostics retain the historical statistic and local null stream.
        # They are not added to form the global statistic. A singleton has none.
        diagnostic = None
        if n >= 2:
            signs = torch.cat(
                (
                    torch.ones(1, n, device=batch_x.time_seqs.device),
                    1
                    - 2
                    * torch.randint(
                        2, (self.n_samples, n), device=batch_x.time_seqs.device
                    ),
                ),
                0,
            )
            values = self.block_statistics(
                histories, batch_x, batch_y, signs, batch_x.time_seqs.device
            )
            diagnostic = self._finish_statistics(values[0], values[1:], False, ids)
        snapshots = tuple(copy_rows(b) for b in (histories, batch_x, batch_y))
        self._batches.append((list(pair_ids), snapshots))
        self._seen_pair_ids.update(ids)
        self._pair_id_mode = True
        self.n_batches += 1
        self._device = batch_x.time_seqs.device
        if self._seed is None:
            self._seed = (torch.initial_seed() + GLOBAL_NULL_RNG_SALT) % (2**63)
        self.total_observed_mmd = self.total_perm_mmds = None
        return diagnostic

    def reset_accumulators(self):
        super().reset_accumulators()
        self._batches.clear()
        self._seed = self._device = None

    def _pooled_observations(self):
        ids = [i for group, _ in self._batches for i in group]
        order = torch.tensor(sorted(range(len(ids)), key=ids.__getitem__))
        result = []
        for column in range(3):
            batches = [b[column] for _, b in self._batches]
            pooled = batches[0]
            for batch in batches[1:]:
                pooled = self._concat_batches(pooled, batch)
            pooled.sequence_ids = ids
            result.append(copy_rows(pooled, order))
        return result

    @torch.no_grad()
    def finalize(self):
        if self.total_observed_mmd is not None or not self._batches:
            return
        histories, x, y = self._pooled_observations()
        n = len(x.time_seqs)
        if n < 2:
            raise ValueError(
                "Global conditional MMD requires at least two independent pairs in total"
            )
        generator = torch.Generator().manual_seed(self._seed)
        signs = torch.cat(
            (
                torch.ones(1, n),
                1 - 2 * torch.randint(2, (self.n_samples, n), generator=generator),
            ),
            0,
        )
        values = self.block_statistics(histories, x, y, signs, self._device)
        self.total_observed_mmd = values[0].cpu()
        self.total_perm_mmds = values[1:].cpu()

    @torch.no_grad()
    def block_statistics(self, histories, x, y, signs, device):
        """Contract every cross-block term on one pooled embedding and swap orbit."""
        n = len(x.time_seqs)
        if self.stopped_event_budget is None and self.observation_horizon is None:
            x, y = self._truncate_to_min_time(x, y)
        pool = self._concat_batches(x, y)
        # Prepare ONCE over the complete swap orbit, before slicing Gram blocks.
        future_paths, _ = self.kernel._prepare_kernel(pool, pool)
        history_paths = None
        if histories.time_seqs.shape[1]:
            history_paths, _ = self.kernel._prepare_kernel(histories, histories)
            history_paths = history_paths.to(device)
        future_paths = future_paths.to(device)
        gram = self.kernel.kernel.compute_Gram
        size = self.block_size
        diagonal = torch.empty(n, dtype=torch.float64, device=device)
        if history_paths is not None:
            for i in range(0, n, size):
                diagonal[i : i + size] = gram(
                    history_paths[i : i + size], history_paths[i : i + size]
                ).diag()
        signs = signs.to(device=device, dtype=torch.float64)
        values = torch.zeros(self.n_samples + 1, dtype=torch.float64, device=device)
        terminal = None
        if self.stopped_event_budget is not None:
            terminal = torch.cat((x.no_next_event_mask, y.no_next_event_mask)).to(
                future_paths
            )
        for i in range(0, n, size):
            a = slice(i, min(i + size, n))
            ai = torch.arange(i, min(i + size, n), device=device)
            for j in range(i, n, size):
                b = slice(j, min(j + size, n))
                bj = torch.arange(j, min(j + size, n), device=device)
                weight = torch.ones(
                    (len(ai), len(bj)), dtype=torch.float64, device=device
                )
                if history_paths is not None:
                    hg = gram(history_paths[a], history_paths[b])
                    weight = torch.exp(
                        -0.5
                        * (diagonal[a, None] + diagonal[None, b] - 2 * hg).clamp_min(0)
                    )
                delta = torch.zeros_like(weight)
                for oi, oj, coefficient in (
                    (0, 0, 1),
                    (n, n, 1),
                    (0, n, -1),
                    (n, 0, -1),
                ):
                    fg = gram(future_paths[ai + oi], future_paths[bj + oj])
                    if terminal is not None:
                        fg *= torch.exp(
                            -0.5
                            * (
                                terminal[ai + oi, None] - terminal[None, bj + oj]
                            ).square()
                        )
                    delta += coefficient * fg
                delta *= weight
                if i == j:
                    delta.fill_diagonal_(0)
                values += (1 if i == j else 2) * torch.einsum(
                    "pi,ij,pj->p", signs[:, a], delta, signs[:, b]
                )
        values /= n * (n - 1)
        if not torch.isfinite(values).all():
            raise ValueError("Cannot finalize nonfinite global statistics")
        return values

    def get_final_p_value(self):
        self.finalize()
        if self.total_observed_mmd is None:
            return torch.tensor(1.0, dtype=torch.float64)
        return monte_carlo_p_value(self.total_observed_mmd, self.total_perm_mmds)

    def global_results(self):
        self.finalize()
        return dict(
            aggregation_method=self.aggregation_method,
            global_observed_statistic=(
                None
                if self.total_observed_mmd is None
                else float(self.total_observed_mmd)
            ),
            global_permuted_statistic=(
                [] if self.total_perm_mmds is None else self.total_perm_mmds.tolist()
            ),
            global_pair_ids=sorted(self._seen_pair_ids),
            global_null_seed=self._seed,
            gram_block_size=self.block_size,
        )
