from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.evaluation.statistical_testing import TestProtocol, MMDTwoSampleTest
from .base_accumulator import BaseAccumulator
from .acc_types import StatisticalMetrics


class StatisticalTestAccumulator(BaseAccumulator):
    """A class to collect and accumulate simulation statistical metrics like
    Maximum Mean Discrepancy (MMD), Kernelized Stein Discrepancy (KSD), and p-values
    for Goodness-of-Fit tests like MMD two-sample tests and Stein-Pangelou tests.
    """

    def __init__(self, mmd_two_sample_test: MMDTwoSampleTest, min_sim_events: int = 1):
        super().__init__(min_sim_events)
        self._mmd_two_sample_test = mmd_two_sample_test
        self._mmd_values: list[float] = []
        self._ksd_values: list[float] = []
        self._mmd_p_values: list[float] = []
        self._ksd_p_values: list[float] = []

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate statistical metrics from batch.

        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)

        """

        # Compute MMD and p-value using the provided MMDTwoSampleTest instance
        mmd_statistic = self._mmd_two_sample_test.get_statistic_from_batches(
            batch, simulation
        )
        mmd_p_value = self._mmd_two_sample_test.p_value_from_batches(batch, simulation)

        # For KSD, we would need a separate SteinTest instance and computation logic
        # Here we just append dummy values for demonstration
        ksd_statistic = 0.0  # Placeholder for actual KSD computation
        ksd_p_value = 1.0  # Placeholder for actual KSD p-value computation

        self._mmd_values.append(mmd_statistic)
        self._mmd_p_values.append(mmd_p_value)
        self._ksd_values.append(ksd_statistic)
        self._ksd_p_values.append(ksd_p_value)

    def compute(self) -> StatisticalMetrics:  # type: ignore[override]
        """Compute final statistics from accumulated data.

        Returns:
            Dictionary containing computed MMD, KSD, and p-value statistics
        """
        return StatisticalMetrics(
            mmd_values=self._mmd_values,
            ksd_values=self._ksd_values,
            mmd_p_values=self._mmd_p_values,
            ksd_p_values=self._ksd_p_values,
        )
