"""
Extractors for simulation metrics with proper batch & mask handling.

Prepares time/type sequences for vectorized metrics computation, including
Wasserstein 1D, Sinkhorn, or other simulation metrics.
"""

from typing import Tuple
from new_ltpp.shared_types import Batch, SimulationResult
from .sim_types import SimTimeValues, SimTypeValues


class SimTimeDataExtractor:
    """Extracts time-related data from simulation batch and predictions, keeping batch structure."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types

    def extract_simulation_time_values(
        self, batch: Batch, sim: SimulationResult
    ) -> SimTimeValues:
        """
        Returns tensors with batch dimension and masks for vectorized metrics:
            true_time_seqs: (B, N)
            true_time_delta_seqs: (B, N)
            sim_time_seqs: (B, N)
            sim_time_delta_seqs: (B, N)
            true_mask: (B, N)
            sim_mask: (B, N)
        """
        return SimTimeValues(
            true_time_seqs=batch.time_seqs,
            true_time_delta_seqs=batch.time_delta_seqs,
            sim_time_seqs=sim.time_seqs,
            sim_time_delta_seqs=sim.dtime_seqs,
            true_mask=batch.seq_non_pad_mask,
            sim_mask=sim.mask,
        )


class SimTypeDataExtractor:
    """Extracts type-related data from simulation batch and predictions, keeping batch structure."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types

    def extract_simulation_type_values(
        self, batch: Batch, sim: SimulationResult
    ) -> SimTypeValues:
        """
        Returns tensors with batch dimension and masks:
            true_type_seqs: (B, N)
            sim_type_seqs: (B, N)
            true_mask: (B, N)
            sim_mask: (B, N)
        """
        return SimTypeValues(
            true_type_seqs=batch.type_seqs,
            sim_type_seqs=sim.type_seqs,
            true_mask=batch.seq_non_pad_mask,
            sim_mask=sim.mask,
        )


class SimulationDataExtractor:
    """
    Unified extractor for simulation batches, keeping batch and mask dimensions
    for vectorized metrics computation.
    """

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
        self.time_extractor = SimTimeDataExtractor(num_event_types)
        self.type_extractor = SimTypeDataExtractor(num_event_types)

    def extract_values(
        self, batch: Batch, sim: SimulationResult
    ) -> Tuple[SimTimeValues, SimTypeValues]:
        time_values = self.time_extractor.extract_simulation_time_values(batch, sim)
        type_values = self.type_extractor.extract_simulation_type_values(batch, sim)
        return time_values, type_values
