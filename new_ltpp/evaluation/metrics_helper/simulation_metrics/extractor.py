"""Extractors for simulation metrics.

Contains time/type extractors and a legacy data extractor to provide the
values expected by the simulation metrics computer.
"""
from typing import Tuple

import torch

from new_ltpp.shared_types import Batch, SimulationResult

from .simul_types import SimulationTimeValues, SimulationTypeValues


class SimulationTimeDataExtractor():
    """Extracts time-related data from simulation batch and predictions."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types

    def extract_simulation_time_values(
        self, batch: Batch, pred: SimulationResult
    ) -> SimulationTimeValues:
        true_time_seqs = batch.time_seqs
        true_time_delta_seqs = batch.time_delta_seqs
        sim_time_seqs = pred.time_seqs
        sim_mask = torch.ones_like(pred.type_seqs, dtype=torch.bool)
        sim_time_delta_seqs = torch.cat([
            sim_time_seqs[:, :1],
            sim_time_seqs[:, 1:] - sim_time_seqs[:, :-1]
        ], dim=1)

        return SimulationTimeValues(
            true_time_seqs=true_time_seqs,
            true_time_delta_seqs=true_time_delta_seqs,
            sim_time_seqs=sim_time_seqs,
            sim_time_delta_seqs=sim_time_delta_seqs,
            sim_mask=sim_mask,
        )


class SimulationTypeDataExtractor():
    """Extracts type-related data from simulation batch and predictions."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types

    def extract_simulation_type_values(self, batch: Batch, pred: SimulationResult) -> SimulationTypeValues:
        sim_type_seqs = pred.type_seqs
        sim_mask = torch.ones_like(sim_type_seqs, dtype=torch.bool)
        return SimulationTypeValues(
            true_type_seqs=batch.type_seqs,
            sim_type_seqs=sim_type_seqs,
            sim_mask=sim_mask,
        )


class SimulationDataExtractor():
    """Legacy extractor providing tuple of tensors expected by the computer."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
        self.time_extractor = SimulationTimeDataExtractor(num_event_types)
        self.type_extractor = SimulationTypeDataExtractor(num_event_types)

    def extract_values(self, batch: Batch, pred: SimulationResult) -> Tuple[torch.Tensor, ...]:
        true_time_seqs = batch.time_seqs
        true_time_delta_seqs = batch.time_delta_seqs
        true_type_seqs = batch.type_seqs

        sim_time_seqs = pred.time_seqs
        sim_type_seqs = pred.type_seqs
        sim_mask = torch.ones_like(sim_type_seqs, dtype=torch.bool)
        sim_time_delta_seqs = torch.cat([
            sim_time_seqs[:, :1],
            sim_time_seqs[:, 1:] - sim_time_seqs[:, :-1]
        ], dim=1)

        return (
            true_time_seqs,
            true_type_seqs,
            true_time_delta_seqs,
            sim_time_seqs,
            sim_type_seqs,
            sim_mask,
        )
