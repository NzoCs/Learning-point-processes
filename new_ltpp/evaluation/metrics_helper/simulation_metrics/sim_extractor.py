"""Extractors for simulation metrics.

Contains time/type extractors and a legacy data extractor to provide the
values expected by the simulation metrics computer.
"""

from typing import Tuple

import torch

from new_ltpp.shared_types import Batch, SimulationResult

from .sim_types import SimTimeValues, SimTypeValues


class SimTimeDataExtractor:
    """Extracts time-related data from simulation batch and predictions."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types

    def extract_simulation_time_values(
        self, batch: Batch, sim: SimulationResult
    ) -> SimTimeValues:

        pad_mask = batch.seq_non_pad_mask
        true_time_seqs = batch.time_seqs[pad_mask]
        true_time_delta_seqs = batch.time_delta_seqs[pad_mask]

        sim_mask = sim.mask
        sim_time_seqs = sim.time_seqs[sim_mask]
        sim_time_delta_seqs = sim.dtime_seqs[sim_mask]

        return SimTimeValues(
            true_time_seqs=true_time_seqs,
            true_time_delta_seqs=true_time_delta_seqs,
            sim_time_seqs=sim_time_seqs,
            sim_time_delta_seqs=sim_time_delta_seqs,
        )


class SimTypeDataExtractor:
    """Extracts type-related data from simulation batch and predictions."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types

    def extract_simulation_type_values(
        self, batch: Batch, sim: SimulationResult
    ) -> SimTypeValues:

        true_type_seqs = batch.type_seqs[batch.seq_non_pad_mask]

        sim_type_seqs = sim.type_seqs[sim.mask]

        return SimTypeValues(
            true_type_seqs=true_type_seqs,
            sim_type_seqs=sim_type_seqs,
        )


class SimulationDataExtractor:
    """Legacy extractor providing tuple of tensors expected by the computer."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
        self.time_extractor = SimTimeDataExtractor(num_event_types)
        self.type_extractor = SimTypeDataExtractor(num_event_types)

    def extract_values(
        self, batch: Batch, sim: SimulationResult
    ) -> Tuple[SimTimeValues, SimTypeValues]:
        time_values = self.time_extractor.extract_simulation_time_values(batch, sim)
        type_values = self.type_extractor.extract_simulation_type_values(batch, sim)

        return (time_values, type_values)
