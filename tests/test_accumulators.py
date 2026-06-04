import numpy as np
import torch
import pytest

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.evaluation.accumulators.mean_len_accumulator import SequenceLengthAccumulator
from new_ltpp.evaluation.accumulators.corr_accumulator import CorrAccumulator


def test_sequence_length_accumulator_edge_cases():
    # 1. Normal batch size = 4
    batch_size = 4
    max_len = 10
    device = torch.device("cpu")

    # Construct batch
    # Sequence lengths: 8, 5, 10, 2
    valid_mask = torch.tensor([
        [True] * 8 + [False] * 2,
        [True] * 5 + [False] * 5,
        [True] * 10,
        [True] * 2 + [False] * 8
    ], dtype=torch.bool, device=device)

    time_seqs = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0],
        [2.0, 4.0, 6.0, 8.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        [5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32, device=device)

    batch = Batch(
        time_seqs=time_seqs,
        time_delta_seqs=torch.zeros_like(time_seqs),
        type_seqs=torch.zeros_like(time_seqs, dtype=torch.long),
        valid_event_mask=valid_mask
    )

    # SimulationResult with normal, 1, and 0 events
    # Seq 0: 5 simulated events
    # Seq 1: 1 simulated event (critical edge case!)
    # Seq 2: 0 simulated events (critical edge case!)
    # Seq 3: 3 simulated events
    sim_mask = torch.tensor([
        [True] * 5 + [False] * 5,
        [True] * 1 + [False] * 9,
        [False] * 10,
        [True] * 3 + [False] * 7
    ], dtype=torch.bool, device=device)

    sim_time_seqs = torch.tensor([
        [1.5, 2.5, 3.5, 4.5, 5.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32, device=device)

    simulation = SimulationResult(
        time_seqs=sim_time_seqs,
        time_delta_seqs=torch.zeros_like(sim_time_seqs),
        type_seqs=torch.zeros_like(sim_time_seqs, dtype=torch.long),
        valid_event_mask=sim_mask
    )

    # Initialize accumulator
    accumulator = SequenceLengthAccumulator(min_sim_events=1)
    accumulator.update(batch, simulation)
    stats = accumulator.compute()

    # Check for NaN and Inf in output arrays and mean/median values
    assert not np.isnan(stats["gt_mean"])
    assert not np.isinf(stats["gt_mean"])
    assert not np.isnan(stats["sim_mean"])
    assert not np.isinf(stats["sim_mean"])

    assert not np.any(np.isnan(stats["gt_array"]))
    assert not np.any(np.isinf(stats["gt_array"]))
    assert not np.any(np.isnan(stats["sim_array"]))
    assert not np.any(np.isinf(stats["sim_array"]))

    print("SequenceLengthAccumulator test passed! No NaNs or Infs found.")


def test_corr_accumulator_edge_cases():
    batch_size = 4
    device = torch.device("cpu")

    # Construct batch
    valid_mask = torch.tensor([
        [True] * 8 + [False] * 2,
        [True] * 5 + [False] * 5,
        [True] * 10,
        [True] * 2 + [False] * 8
    ], dtype=torch.bool, device=device)

    time_seqs = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0],
        [2.0, 4.0, 6.0, 8.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        [5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32, device=device)

    batch = Batch(
        time_seqs=time_seqs,
        time_delta_seqs=torch.zeros_like(time_seqs),
        type_seqs=torch.zeros_like(time_seqs, dtype=torch.long),
        valid_event_mask=valid_mask
    )

    # SimulationResult with normal, 1, and 0 events
    sim_mask = torch.tensor([
        [True] * 5 + [False] * 5,
        [True] * 1 + [False] * 9,
        [False] * 10,
        [True] * 3 + [False] * 7
    ], dtype=torch.bool, device=device)

    sim_time_seqs = torch.tensor([
        [1.5, 2.5, 3.5, 4.5, 5.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32, device=device)

    simulation = SimulationResult(
        time_seqs=sim_time_seqs,
        time_delta_seqs=torch.zeros_like(sim_time_seqs),
        type_seqs=torch.zeros_like(sim_time_seqs, dtype=torch.long),
        valid_event_mask=sim_mask
    )

    accumulator = CorrAccumulator(min_sim_events=1, nb_bins=10, max_lag=5)
    accumulator.update(batch, simulation)
    stats = accumulator.compute()

    # Check for NaN or Inf in the correlation averages
    assert not np.any(np.isnan(stats["acf_gt_mean"]))
    assert not np.any(np.isinf(stats["acf_gt_mean"]))
    assert not np.any(np.isnan(stats["acf_sim_mean"]))
    assert not np.any(np.isinf(stats["acf_sim_mean"]))

    print("CorrAccumulator test passed! No NaNs or Infs found in ACF.")
