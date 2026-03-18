from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.evaluation.statistical_testing.statistical_tests.builder import (
    StatisticalTestDict,
    StatisticalTestBuilder,
)
from .base_accumulator import Accumulator
from .acc_types import StatisticalMetrics


class StatisticalTestAccumulator(Accumulator):
    """A class to collect and accumulate simulation statistical metrics like
    Maximum Mean Discrepancy (MMD), Kernelized Stein Discrepancy (KSD), and p-values
    for Goodness-of-Fit tests like MMD two-sample tests and Stein-Pangelou tests.
    """

    def __init__(
        self,
        statistical_test_config: StatisticalTestDict,
        min_sim_events: int = 1,
    ):
        super().__init__(min_sim_events)
        self.statistical_test = (
            StatisticalTestBuilder().from_dict(statistical_test_config).build()
        )
        self.statistics: list[float] = []
        self.p_values: list[float] = []
        self.perm_statistics: list[float] = []

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate statistical metrics from batch.

        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)

        """

        # Compute MMD and p-value using the provided MMDTwoSampleTest instance
        p_value, statistic, perm_statistic = (
            self.statistical_test.statistics_from_batches(batch, simulation)
        )

        self.statistics.append(statistic.item())
        self.p_values.append(p_value.item())
        self.perm_statistics.extend(perm_statistic.tolist())

    def compute(self) -> StatisticalMetrics:  # type: ignore[override]
        """Compute final statistics from accumulated data.

        Returns:
            Dictionary containing computed MMD, and p-value statistics
        """
        return StatisticalMetrics(
            mmd_values=self.statistics,
            mmd_p_values=self.p_values,
            mmd_perm_distributions=self.perm_statistics,
        )
