from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.evaluation.statistical_testing import MMDTwoSampleTest
from .base_accumulator import Accumulator
from .acc_types import StatisticalMetrics


class StatisticalTestAccumulator(Accumulator):
    """A class to collect and accumulate simulation statistical metrics like
    Maximum Mean Discrepancy (MMD), Kernelized Stein Discrepancy (KSD), and p-values
    for Goodness-of-Fit tests like MMD two-sample tests and Stein-Pangelou tests.
    """

    def __init__(self, mmd_two_sample_test: MMDTwoSampleTest, min_sim_events: int = 1):
        super().__init__(min_sim_events)
        self._mmd_two_sample_test = mmd_two_sample_test
        self._mmd_values: list[float] = []
        self._mmd_p_values: list[float] = []
        self._mmd_perm_distributions: list[float] = []

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate statistical metrics from batch.

        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)

        """

        # Compute MMD and p-value using the provided MMDTwoSampleTest instance
        mmd_p_value, mmd_statistic, perm_mmds = (
            self._mmd_two_sample_test.statistics_from_batches(batch, simulation)
        )

        self._mmd_values.append(mmd_statistic.item())
        self._mmd_p_values.append(mmd_p_value.item())
        self._mmd_perm_distributions.extend(perm_mmds.tolist())

    def compute(self) -> StatisticalMetrics:  # type: ignore[override]
        """Compute final statistics from accumulated data.

        Returns:
            Dictionary containing computed MMD, and p-value statistics
        """
        return StatisticalMetrics(
            mmd_values=self._mmd_values,
            mmd_p_values=self._mmd_p_values,
            mmd_perm_distributions=self._mmd_perm_distributions,
        )
