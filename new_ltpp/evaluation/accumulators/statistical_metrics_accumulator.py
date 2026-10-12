import torch

from new_ltpp.configs.statistical_test_config import StatisticalTestConfig
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    create_statistical_test,
)
from new_ltpp.shared_types import Batch, SimulationResult

from .acc_types import StatisticalTestData
from .base_accumulator import Accumulator

MMD_NULL_RNG_SALT = 0x4D4D44


class StatisticalTestAccumulator(Accumulator):
    """A class to collect and accumulate simulation statistical metrics like
    Maximum Mean Discrepancy (MMD), Kernelized Stein Discrepancy (KSD), and p-values
    for Goodness-of-Fit tests like MMD two-sample tests and Stein-Pangelou tests.
    """

    def __init__(
        self,
        statistical_test_config: StatisticalTestConfig,
        min_sim_events: int = 1,
        *,
        conditional: bool = False,
    ):
        super().__init__(min_sim_events)

        self.test = create_statistical_test(statistical_test_config)
        self.conditional = conditional
        if conditional:
            from new_ltpp.evaluation.statistical_testing.statistical_tests import (
                ConditionalMMDTwoSampleTest,
                MMDTwoSampleTest,
            )

            if not isinstance(self.test, MMDTwoSampleTest):
                raise ValueError("Conditional resampling is implemented for MMD only")
            self.test = ConditionalMMDTwoSampleTest(
                self.test.kernel, self.test.n_samples
            )
        self.p_values: list[float] = []
        self.observed_statistic: list[float] = []
        self.perm_statistics: list[float] = []
        self.num_sequences: int = 0
        self.null_sample_counts: list[int] = []
        self.null_methods: list[str] = []
        self._null_rng_states: dict[str, torch.Tensor] = {}

    def update(
        self,
        batch: Batch,
        simulation: SimulationResult,
        *,
        histories: Batch | None = None,
        pair_ids=None,
    ) -> None:
        """Accumulate statistical metrics from batch.

        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)

        """
        # Statistical resampling must not alter the generator used for trajectories.
        if self.conditional != (histories is not None):
            raise ValueError(
                "Conditional mode requires shared histories; unconditional mode must not receive them"
            )
        devices = [batch.time_seqs.device] if batch.time_seqs.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            # Separate persistent streams avoid reusing the trajectory draws.
            seed = (torch.initial_seed() + MMD_NULL_RNG_SALT) % (2**63)
            cpu_state = self._null_rng_states.get("cpu")
            if cpu_state is None:
                cpu_state = torch.Generator().manual_seed(seed).get_state()
            torch.set_rng_state(cpu_state)
            if batch.time_seqs.is_cuda:
                device = batch.time_seqs.device
                key = str(device)
                cuda_state = self._null_rng_states.get(key)
                if cuda_state is None:
                    cuda_state = (
                        torch.Generator(device=device).manual_seed(seed).get_state()
                    )
                torch.cuda.set_rng_state(cuda_state, device)
            if self.conditional:
                stats = self.test.compute_statistics(
                    batch, simulation, histories=histories, pair_ids=pair_ids
                )
            else:
                if pair_ids is not None:
                    stats = self.test.compute_statistics(
                        batch, simulation, pair_ids=pair_ids
                    )
                else:
                    stats = self.test.compute_statistics(batch, simulation)
            self._null_rng_states["cpu"] = torch.get_rng_state()
            if batch.time_seqs.is_cuda:
                self._null_rng_states[str(batch.time_seqs.device)] = (
                    torch.cuda.get_rng_state(batch.time_seqs.device)
                )

        self.null_sample_counts.append(stats["num_null_samples"])
        self.null_methods.append(stats["null_method"])
        self.p_values.append(stats["p_value"].item())
        self.observed_statistic.append(stats["observed_statistic"].item())
        self.perm_statistics.extend(stats["permuted_statistics"].tolist())
        self.num_sequences += batch.time_seqs.shape[0]

    def reset(self) -> None:
        """Reset accumulated statistical test data."""
        super().reset()
        self.p_values = []
        self.observed_statistic = []
        self.perm_statistics = []
        self.num_sequences = 0
        self.null_sample_counts = []
        self.null_methods = []
        self._null_rng_states.clear()
        if hasattr(self.test, "reset_accumulators"):
            self.test.reset_accumulators()

    def compute(self) -> StatisticalTestData:  # type: ignore[override]
        """Compute final statistics from accumulated data.

        Returns:
            Dictionary containing computed MMD, and p-value statistics
        """
        pooled_p_val = None
        if hasattr(self.test, "get_final_p_value"):
            try:
                pooled_p_val = self.test.get_final_p_value().item()
            except Exception:
                pass

        return StatisticalTestData(
            p_values=self.p_values,
            observed_statistic=self.observed_statistic,
            permuted_statistic=self.perm_statistics,
            pooled_p_value=pooled_p_val,
            num_sequences=self.num_sequences,
            null_sample_counts=self.null_sample_counts,
            null_methods=self.null_methods,
        )
