from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.evaluation.statistical_testing.statistical_tests.builder import (
    StatisticalTestConfig,
    StatisticalTestBuilder,
)
from .base_accumulator import Accumulator
from .acc_types import StatisticalTestData


class StatisticalTestAccumulator(Accumulator):
    """A class to collect and accumulate simulation statistical metrics like
    Maximum Mean Discrepancy (MMD), Kernelized Stein Discrepancy (KSD), and p-values
    for Goodness-of-Fit tests like MMD two-sample tests and Stein-Pangelou tests.
    """

    def __init__(
        self,
        statistical_test_config: StatisticalTestConfig,
        min_sim_events: int = 1,
    ):
        super().__init__(min_sim_events)
        self.statistical_test = (
            StatisticalTestBuilder().from_dict(statistical_test_config).build()
        )
        self.p_values: list[float] = []
        self.observed_statistic: list[float] = []
        self.perm_statistics: list[float] = []

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate statistical metrics from batch.

        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)

        """

        # Compute MMD and p-value using the provided MMDTwoSampleTest instance
        stats = self.statistical_test.compute_statistics(batch, simulation)

        self.p_values.append(stats["p_value"].item())
        self.observed_statistic.append(stats["observed_statistic"].item())
        self.perm_statistics.extend(stats["permuted_statistics"].tolist())

    def compute(self) -> StatisticalTestData:  # type: ignore[override]
        """Compute final statistics from accumulated data.

        Returns:
            Dictionary containing computed MMD, and p-value statistics
        """
        return StatisticalTestData(
            p_values=self.p_values,
            observed_statistic=self.observed_statistic,
            permuted_statistic=self.perm_statistics,
        )
