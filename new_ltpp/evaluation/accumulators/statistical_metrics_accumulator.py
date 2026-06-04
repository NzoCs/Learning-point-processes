from typing import Optional, Any
from new_ltpp.evaluation.statistical_testing.statistical_tests import create_statistical_test
from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.configs.statistical_test_config import StatisticalTestConfig
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
        simulator: Optional[Any] = None,
    ):
        super().__init__(min_sim_events)

        self.test = create_statistical_test(statistical_test_config)
        self.simulator = simulator
        self.p_values: list[float] = []
        self.observed_statistic: list[float] = []
        self.perm_statistics: list[float] = []
        self.num_sequences: int = 0

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Accumulate statistical metrics from batch.

        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)

        """
        from typing import Any
        simulations = [simulation]
        if self.simulator is not None and hasattr(self.test, "n_samples"):
            n_samples = self.test.n_samples
            if n_samples > 1:
                N = n_samples - 1
                B = batch.time_seqs.shape[0]
                
                # Stack the input batch N times along the batch dimension
                batch_stacked = Batch(
                    time_seqs=batch.time_seqs.repeat(N, 1),
                    time_delta_seqs=batch.time_delta_seqs.repeat(N, 1),
                    type_seqs=batch.type_seqs.repeat(N, 1),
                    valid_event_mask=batch.valid_event_mask.repeat(N, 1),
                )
                
                # Run the simulator in parallel on the GPU
                sim_stacked = self.simulator.simulate(batch_stacked)
                
                # Split the stacked simulation results back into N individual batches
                time_seqs_list = sim_stacked.time_seqs.split(B, dim=0)
                time_delta_seqs_list = sim_stacked.time_delta_seqs.split(B, dim=0)
                type_seqs_list = sim_stacked.type_seqs.split(B, dim=0)
                valid_event_mask_list = sim_stacked.valid_event_mask.split(B, dim=0)
                
                for i in range(N):
                    simulations.append(
                        Batch(
                            time_seqs=time_seqs_list[i],
                            time_delta_seqs=time_delta_seqs_list[i],
                            type_seqs=type_seqs_list[i],
                            valid_event_mask=valid_event_mask_list[i],
                        )
                    )

        # Compute MMD and p-value using the provided simulations
        stats = self.test.compute_statistics(batch, simulation, simulations=simulations)

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
        )
